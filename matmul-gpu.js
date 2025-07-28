window.bufferTimes = [];
async function createOperations(device) {
    function createPipeline(code, bindGroupTypes) {
        return device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [
                    device.createBindGroupLayout({
                        entries: bindGroupTypes.map((type, i) => ({
                            binding: i,
                            visibility: GPUShaderStage.COMPUTE,
                            buffer: { type },
                        })),
                    })
                ],
            }),
            compute: {
                module: device.createShaderModule({ code }),
                entryPoint: 'main',
            },
        });
    }

    const codeFasterMatMul = await fetch('../faster-matmul.wgsl').then(response => response.text());
    const codeMatmul = await fetch('../matmul.wgsl').then(response => response.text());
    const codeScatterAdd = await fetch('../scatter-add.wgsl').then(response => response.text());
    const codeMatmulBatch = await fetch('../matmul-batch.wgsl').then(response => response.text());
    const codeBatchSoftmaxRowTril = await fetch('../batch-softmax-row-tril.wgsl').then(response => response.text());
    const codeBatchSoftmaxRowTrilBackward = await fetch('../batch-softmax-row-tril--backward.wgsl').then(response => response.text());
    const codeReLU = await fetch('../relu.wgsl').then(response => response.text());
    const codeBiasGradientAccumulation = await fetch('../bias-gradient-accumulation.wgsl').then(response => response.text());

    const computePipelineMatmul = createPipeline(codeMatmul, ['uniform', 'storage', 'read-only-storage', 'read-only-storage', 'read-only-storage']);
    const computePipelineFasterMatMul = createPipeline(codeFasterMatMul, ['uniform', 'storage', 'read-only-storage', 'read-only-storage', 'read-only-storage']);
    const computePipelineMatmulBatch = createPipeline(codeMatmulBatch, ['uniform', 'read-only-storage', 'read-only-storage', 'storage']);
    const computePipelineScatterAdd = createPipeline(codeScatterAdd, ['uniform', 'storage', 'read-only-storage', 'read-only-storage']);
    const computePipelineBatchSoftmaxRowTril = createPipeline(codeBatchSoftmaxRowTril, ['uniform', 'storage']);
    const computePipelineBatchSoftmaxRowTrilBackward = createPipeline(codeBatchSoftmaxRowTrilBackward, ['uniform', 'read-only-storage', 'read-only-storage', 'storage']);
    const computePipelineReLU = createPipeline(codeReLU, ['uniform', 'storage']);
    const computePipelineBiasGradientAccumulation = createPipeline(codeBiasGradientAccumulation, ['uniform', 'read-only-storage', 'storage']);

    function copyToCPU(commandEncoder, buffer) {
        const readBuffer = device.createBuffer({
            size: buffer.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });
        commandEncoder.copyBufferToBuffer(buffer, 0, readBuffer, 0, buffer.size);
        return async () => {
            await readBuffer.mapAsync(GPUMapMode.READ);
            const result = readBuffer.getMappedRange().slice(0);
            // readBuffer.unmap();
            readBuffer.destroy();
            return result;
        }
    }

    /**
     * @param {Float32Array} A - Matrix A data (shape: [M, K])
     * @param {Float32Array} B - Matrix B data (shape: [K, N])
     * @param {number} aT - transpose flag for A (0 or 1)
     * @param {number} bT - transpose flag for B (0 or 1)
     */
    async function mm(A, B, aT = 0, bT = 0, bias) {
        const M = aT ? A.shape[1] : A.shape[0];
        const N = bT ? B.shape[0] : B.shape[1];
        const K = aT ? A.shape[0] : A.shape[1];
        if (K !== (bT ? B.shape[1] : B.shape[0])) {
            throw new Error('Matrix dimensions do not match.');
        }

        if (bias) {
            if (N !== bias.length) {
                throw new Error('Bias vector dimension does not match the resulting matrix rows.');
            }
        }

        const start = performance.now();

        // Pass transpose flags as uint32 in uniform buffer
        const uniformArray = new Uint32Array([M, N, K, aT, bT]);
        const uniformBuffer = device.createBuffer({
            size: uniformArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

        const outputBuffer = device.createBuffer({
            size: new Float32Array(M * N).byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        const aBuffer = device.createBuffer({
            size: A.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bBuffer = device.createBuffer({
            size: B.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const biasBuffer = device.createBuffer({
            size: bias ? bias.byteLength : 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(aBuffer, 0, A);
        device.queue.writeBuffer(bBuffer, 0, B);
        if (bias) {
            device.queue.writeBuffer(biasBuffer, 0, bias);
        }

        const bindGroup0 = device.createBindGroup({
            layout: computePipelineMatmul.getBindGroupLayout(0),
            entries: [uniformBuffer, outputBuffer, aBuffer, bBuffer, biasBuffer].map((buffer, i) => ({
                binding: i,
                resource: { buffer },
            })),
        });

        const workgroupSize = 16;
        const workgroupsX = Math.ceil(N / workgroupSize);
        const workgroupsY = Math.ceil(M / workgroupSize);
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const passEncoder = commandEncoder.beginComputePass();

        passEncoder.setPipeline(computePipelineMatmul);
        passEncoder.setBindGroup(0, bindGroup0);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        passEncoder.end();

        const readBuffer = copyToCPU(commandEncoder, outputBuffer);
        device.queue.submit([commandEncoder.finish()]);
        [uniformBuffer, outputBuffer, aBuffer, bBuffer, biasBuffer].forEach(buffer => buffer.destroy());
        return new FloatMatrix(await readBuffer()).reshape([M, N]);
    };

    async function fasterMatMul(A, B, bias) {
        const M = A.shape[0];
        const N = B.shape[1];
        const K = A.shape[1];
        if (K !== B.shape[0]) {
            throw new Error('Matrix dimensions do not match.');
        }

        if (bias) {
            if (N !== bias.length) {
                throw new Error('Bias vector dimension does not match the resulting matrix rows.');
            }
        }

        if ( 4 % N !== 0 || 4 % K !== 0 ) {
            return await mm(A, B, 0, 0, bias);
        }

        const start = performance.now();

        const ND4 = Math.ceil(N / 4);
        const KD4 = Math.ceil(K / 4);
        const uniformArray = new Uint32Array([M, N, ND4, KD4]);
        const uniformBuffer = device.createBuffer({
            size: uniformArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformArray);

        const outputBuffer = device.createBuffer({
            size: new Float32Array(M * ND4 * 4).byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        const aBuffer = device.createBuffer({
            size: A.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bBuffer = device.createBuffer({
            size: B.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const biasBuffer = device.createBuffer({
            size: bias ? ND4 * 4 * 4 : 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(aBuffer, 0, A);
        device.queue.writeBuffer(bBuffer, 0, B);
        if (bias) {
            device.queue.writeBuffer(biasBuffer, 0, bias);
        }

        const bindGroup0 = device.createBindGroup({
            layout: computePipelineMatmul.getBindGroupLayout(0),
            entries: [uniformBuffer, outputBuffer, aBuffer, bBuffer, biasBuffer].map((buffer, i) => ({
                binding: i,
                resource: { buffer },
            })),
        });

        const tileWidth = 2 * 4;  // 8 floats per thread horizontally (2 vec4 columns * 4 floats)
        const tileHeight = 4;     // 4 rows per thread vertically
        const workgroupSizeX = 8; // as per @workgroup_size(8,8)
        const workgroupSizeY = 8;
        const workgroupsX = Math.ceil(N / (tileWidth * workgroupSizeX)); 
        const workgroupsY = Math.ceil(M / (tileHeight * workgroupSizeY));
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const passEncoder = commandEncoder.beginComputePass();

        passEncoder.setPipeline(computePipelineFasterMatMul);
        passEncoder.setBindGroup(0, bindGroup0);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        passEncoder.end();

        const readBuffer = copyToCPU(commandEncoder, outputBuffer);
        device.queue.submit([commandEncoder.finish()]);
        [uniformBuffer, outputBuffer, aBuffer, bBuffer, biasBuffer].forEach(buffer => buffer.destroy());
        return new FloatMatrix(await readBuffer()).reshape([M, N]);
    };

    async function batchMatMul(A, B, aT = 0, bT = 0) {
        const BATCH = A.shape[0];
        const M = aT ? A.shape[2] : A.shape[1];
        const K = aT ? A.shape[1] : A.shape[2];
        const N = bT ? B.shape[1] : B.shape[2];
        if (K !== (bT ? B.shape[2] : B.shape[1]) || BATCH !== B.shape[0]) {
            throw new Error('Matrix dimensions do not match.');
        }

        const start = performance.now();

        const uniformArray = new Uint32Array([BATCH, M, N, K, aT, bT]);
        const uniformBuffer = device.createBuffer({
            size: uniformArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
    
        const aBuffer = device.createBuffer({
            size: A.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const bBuffer = device.createBuffer({
            size: B.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const outputBuffer = device.createBuffer({
            size: 4 * BATCH * M * N,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
    
        device.queue.writeBuffer(aBuffer, 0, A);
        device.queue.writeBuffer(bBuffer, 0, B);
    
        const bindGroup = device.createBindGroup({
            layout: computePipelineMatmulBatch.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: aBuffer } },
                { binding: 2, resource: { buffer: bBuffer } },
                { binding: 3, resource: { buffer: outputBuffer } },
            ]
        });
    
        const workgroupSize = 8;
        const workgroupsX = Math.ceil(N / workgroupSize);
        const workgroupsY = Math.ceil(M / workgroupSize);
        const workgroupsZ = BATCH;
    
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipelineMatmulBatch);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
        passEncoder.end();
    
        const readBuffer = copyToCPU(commandEncoder, outputBuffer);
        device.queue.submit([commandEncoder.finish()]);
        [aBuffer, bBuffer, uniformBuffer, outputBuffer].forEach(buf => buf.destroy());
        return new FloatMatrix(await readBuffer()).reshape([BATCH, M, N]);
    }

    async function scatterAdd(grad, indices, shape) {
        const B_len = grad.length / shape[1]; // total rows in grad
        const Dim = shape[1];
        const M = shape[0];

        const start = performance.now();
      
        // Uniform: Dim and total length for dispatching
        const uniformArray = new Uint32Array([Dim, B_len]);
        const uniformBuffer = device.createBuffer({
            size: uniformArray.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformArray);
      
        // Output buffer (uninitialized, GPU will zero in shader or we rely on atomicAdd correctness)
        const outputBuffer = device.createBuffer({
            size: new Float32Array(M * Dim).byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        const gradBuffer = device.createBuffer({
            size: grad.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        const indicesBuffer = device.createBuffer({
            size: new Int32Array(indices).byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(gradBuffer, 0, grad);
        device.queue.writeBuffer(indicesBuffer, 0, new Int32Array(indices));

        const bindGroup = device.createBindGroup({
          layout: computePipelineScatterAdd.getBindGroupLayout(0),
          entries: [uniformBuffer, outputBuffer, gradBuffer, indicesBuffer].map((buffer, i) => ({
            binding: i,
            resource: { buffer },
          })),
        });

        const workgroupSize = 64;
        const workgroupsX = Math.ceil(B_len / workgroupSize);
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const passEncoder = commandEncoder.beginComputePass();

        passEncoder.setPipeline(computePipelineScatterAdd);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(workgroupsX);
        passEncoder.end();

        const readBuffer = copyToCPU(commandEncoder, outputBuffer);
        device.queue.submit([commandEncoder.finish()]);
        [uniformBuffer, outputBuffer, gradBuffer, indicesBuffer].forEach(buffer => buffer.destroy());
        return new FloatMatrix(await readBuffer()).reshape(shape);
    }

    async function batchSoftmaxRowTril(A) {
        const start = performance.now();
        const [B, T] = A.shape;
        // Uniform buffer for dims [B, T]
        const uniformData = new Uint32Array([B, T]);
        const uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Storage buffer for matrix data
        const bufferSize = A.byteLength;
        const storageBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Float32Array(storageBuffer.getMappedRange()).set(A);
        storageBuffer.unmap();

        // Bind group
        const bindGroup = device.createBindGroup({
            layout: computePipelineBatchSoftmaxRowTril.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: storageBuffer } },
            ],
        });

        // Command encoder
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(computePipelineBatchSoftmaxRowTril);
        pass.setBindGroup(0, bindGroup);
        const wgSize = 16;
        pass.dispatchWorkgroups(
            Math.ceil(T / wgSize),
            Math.ceil(T / wgSize),
            B
        );
        pass.end();

        // Submit
        device.queue.submit([commandEncoder.finish()]);

        const readEncoder = device.createCommandEncoder();
        const readBuffer = copyToCPU(readEncoder, storageBuffer);
        device.queue.submit([readEncoder.finish()]);

        const output = new FloatMatrix(await readBuffer()).reshape([B, T, T]);

        [uniformBuffer, storageBuffer].forEach(buffer => buffer.destroy()); 

        return output;
    }

    async function batchSoftmaxRowTrilBackward(dOut, Out) {
        const start = performance.now();
        const [B, T] = dOut.shape;
    
        // Uniform buffer for dims [B, T]
        const uniformData = new Uint32Array([B, T]);
        const uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Storage buffers
        const bufferSize = dOut.byteLength;
    
        // dOut buffer
        const dOutBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(dOutBuffer.getMappedRange()).set(dOut);
        dOutBuffer.unmap();
    
        // Out buffer
        const outBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(outBuffer.getMappedRange()).set(Out);
        outBuffer.unmap();
    
        // dIn output buffer
        const dInBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
        });
    
        // Bind group
        const bindGroup = device.createBindGroup({
            layout: computePipelineBatchSoftmaxRowTrilBackward.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: outBuffer } },
                { binding: 2, resource: { buffer: dOutBuffer } },
                { binding: 3, resource: { buffer: dInBuffer } },
            ],
        });
    
        // Dispatch
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(computePipelineBatchSoftmaxRowTrilBackward);
        pass.setBindGroup(0, bindGroup);
        const wgSize = 16;
        pass.dispatchWorkgroups(
            Math.ceil(T / wgSize),
            Math.ceil(T / wgSize),
            B
        );
        pass.end();
    
        device.queue.submit([commandEncoder.finish()]);
    
        // Read result
        const readEncoder = device.createCommandEncoder();
        const readBuffer = copyToCPU(readEncoder, dInBuffer);
        device.queue.submit([readEncoder.finish()]);

        [uniformBuffer, outBuffer, dOutBuffer, dInBuffer].forEach(buffer => buffer.destroy());
    
        const dIn = new FloatMatrix(await readBuffer()).reshape([B, T, T]);
    
        return dIn;
    }

    async function relu(A) {
        const start = performance.now();
        // Uniform buffer for dims [B, T]
        const uniformData = new Uint32Array([A.length]);
        const uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);

        // Storage buffer for matrix data
        const bufferSize = A.byteLength;
        const storageBuffer = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Float32Array(storageBuffer.getMappedRange()).set(A);
        storageBuffer.unmap();

        // Bind group
        const bindGroup = device.createBindGroup({
            layout: computePipelineReLU.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: storageBuffer } },
            ],
        });

        const maxWorkgroupsX = 65535;
        const workgroupsX = Math.min(A.length, maxWorkgroupsX);
        const workgroupsY = Math.ceil(A.length / maxWorkgroupsX);

        // Command encoder
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(computePipelineReLU);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY);
        pass.end();

        // Submit
        device.queue.submit([commandEncoder.finish()]);

        const readEncoder = device.createCommandEncoder();
        const readBuffer = copyToCPU(readEncoder, storageBuffer);
        device.queue.submit([readEncoder.finish()]);

        [uniformBuffer, storageBuffer].forEach(buffer => buffer.destroy());

        const output = new FloatMatrix(await readBuffer()).reshape(A.shape);

        return output;
    }

    async function biasGradSum(grad, m, n) {
        const start = performance.now();
        // dims uniform buffer [m, n]
        const uniformData = new Uint32Array([m, n]);
        const uniformBuffer = device.createBuffer({
            size: uniformData.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    
        // Storage buffer for grad (input)
        const gradBuffer = device.createBuffer({
            size: grad.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(gradBuffer.getMappedRange()).set(grad);
        gradBuffer.unmap();
    
        // Storage buffer for biasGrad (output)
        const biasGradBuffer = device.createBuffer({
            size: n * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
    
        const bindGroup = device.createBindGroup({
            layout: computePipelineBiasGradientAccumulation.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: uniformBuffer } },
                { binding: 1, resource: { buffer: gradBuffer } },
                { binding: 2, resource: { buffer: biasGradBuffer } },
            ],
        });
    
        // Command encoder & compute pass
        const commandEncoder = device.createCommandEncoder();
        window.bufferTimes.push(performance.now() - start);
        const pass = commandEncoder.beginComputePass();
        pass.setPipeline(computePipelineBiasGradientAccumulation);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(n / 64));
        pass.end();
    
        device.queue.submit([commandEncoder.finish()]);
    
        // Read back biasGrad results
        const readEncoder = device.createCommandEncoder();
        const readBuffer = copyToCPU(readEncoder, biasGradBuffer);
        device.queue.submit([readEncoder.finish()]);

        [uniformBuffer, gradBuffer, biasGradBuffer].forEach(buffer => buffer.destroy());
    
        return new FloatMatrix(await readBuffer()).reshape([n]);
    }
    
    return {
        matMul: mm,
        fasterMatMul,
        scatterAdd,
        batchMatMul,
        batchSoftmaxRowTril,
        batchSoftmaxRowTrilBackward,
        relu,
        biasGradSum
    };
}

export async function GPU() {
    if (!navigator.gpu) {
        throw new Error('GPU not supported. Try using Chrome.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    return await createOperations(device);
}
