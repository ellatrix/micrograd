async function createOperations(device) {
    const code = await fetch('../matmul.wgsl').then(response => response.text());
    const codeScatterAdd = await fetch('../scatter-add.wgsl').then(response => response.text());
    const bindGroupLayoutMatMul0 = device.createBindGroupLayout({
        entries: ['uniform', 'storage', 'read-only-storage', 'read-only-storage', 'read-only-storage'].map((type, i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type },
        })),
    });
    const bindGroupLayoutScatterAdd0 = device.createBindGroupLayout({
        entries: ['uniform', 'storage', 'read-only-storage', 'read-only-storage'].map((type, i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type },
        })),
    });

    const pipelineLayoutMatMul = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayoutMatMul0],
    });
    const pipelineLayoutScatterAdd = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayoutScatterAdd0],
    });

    const codeMatmul = code.replace('/*INDEXA*/', 'y * K + k').replace('/*INDEXB*/', 'k * N + x');
    const codeMatmulTransposeA = code.replace('/*INDEXA*/', 'y + k * M').replace('/*INDEXB*/', 'k * N + x');
    const codeMatmulTransposeB = code.replace('/*INDEXA*/', 'y * K + k').replace('/*INDEXB*/', 'x * K + k');
    const codeMatmulTransposeAB = code.replace('/*INDEXA*/', 'y + k * M').replace('/*INDEXB*/', 'x * K + k');

    const computePipelineMatmul = device.createComputePipeline({
        layout: pipelineLayoutMatMul,
        compute: {
            module: device.createShaderModule({ code: codeMatmul }),
            entryPoint: 'main',
        },
    });

    const computePipelineMatmulTransposeA = device.createComputePipeline({
        layout: pipelineLayoutMatMul,
        compute: {
            module: device.createShaderModule({ code: codeMatmulTransposeA }),
            entryPoint: 'main',
        },
    });

    const computePipelineMatmulTransposeB = device.createComputePipeline({
        layout: pipelineLayoutMatMul,
        compute: {
            module: device.createShaderModule({ code: codeMatmulTransposeB }),
            entryPoint: 'main',
        },
    });

    const computePipelineMatmulTransposeAB = device.createComputePipeline({
        layout: pipelineLayoutMatMul,
        compute: {
            module: device.createShaderModule({ code: codeMatmulTransposeAB }),
            entryPoint: 'main',
        },
    });

    const matMulPipelineMap = [
        [computePipelineMatmul, computePipelineMatmulTransposeB],
        [computePipelineMatmulTransposeA, computePipelineMatmulTransposeAB],
    ];

    const computePipelineScatterAdd = device.createComputePipeline({
        layout: pipelineLayoutScatterAdd,
        compute: {
            module: device.createShaderModule({ code: codeScatterAdd }),
            entryPoint: 'main',
        },
    });

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

        // Pass transpose flags as uint32 in uniform buffer
        const uniformArray = new Uint32Array([M, N, K]);
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
            layout: bindGroupLayoutMatMul0,
            entries: [uniformBuffer, outputBuffer, aBuffer, bBuffer, biasBuffer].map((buffer, i) => ({
                binding: i,
                resource: { buffer },
            })),
        });

        const workgroupSize = 16;
        const workgroupsX = Math.ceil(N / workgroupSize);
        const workgroupsY = Math.ceil(M / workgroupSize);
        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();

        passEncoder.setPipeline(matMulPipelineMap[Number(aT)][Number(bT)]);
        passEncoder.setBindGroup(0, bindGroup0);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        passEncoder.end();

        const readBuffer = copyToCPU(commandEncoder, outputBuffer);
        device.queue.submit([commandEncoder.finish()]);
        [uniformBuffer, outputBuffer, aBuffer, bBuffer, biasBuffer].forEach(buffer => buffer.destroy());
        return new FloatMatrix(await readBuffer()).reshape([M, N]);
    };

    async function scatterAdd(grad, indices, shape) {
        const B_len = grad.length / shape[1]; // total rows in grad
        const Dim = shape[1];
        const M = shape[0];
      
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
          layout: bindGroupLayoutScatterAdd0,
          entries: [uniformBuffer, outputBuffer, gradBuffer, indicesBuffer].map((buffer, i) => ({
            binding: i,
            resource: { buffer },
          })),
        });

        const workgroupSize = 64;
        const workgroupsX = Math.ceil(B_len / workgroupSize);
        const commandEncoder = device.createCommandEncoder();
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

    return { matMul: mm, scatterAdd };
}

export async function GPU() {
    if (!navigator.gpu) {
        throw new Error('GPU not supported. Try using Chrome.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    return await createOperations(device);
}
