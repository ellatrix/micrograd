const bufferPool = new Set();

async function createMatMul(device) {
    const code = await fetch('matmul.wgsl').then(response => response.text());
    const bindGroupLayout0 = device.createBindGroupLayout({
        entries: ['uniform', 'storage', 'read-only-storage', 'read-only-storage'].map((type, i) => ({
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type },
        })),
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout0],
    });

    const codeMatmul = code.replace('/*INDEXA*/', 'y * K + k').replace('/*INDEXB*/', 'k * N + x');
    const codeMatmulTransposeA = code.replace('/*INDEXA*/', 'y + k * M').replace('/*INDEXB*/', 'k * N + x');
    const codeMatmulTransposeB = code.replace('/*INDEXA*/', 'y * K + k').replace('/*INDEXB*/', 'x * K + k');
    const codeMatmulTransposeAB = code.replace('/*INDEXA*/', 'y + k * M').replace('/*INDEXB*/', 'x * K + k');

    const computePipelineMatmul = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: device.createShaderModule({ code: codeMatmul }),
            entryPoint: 'main',
        },
    });

    const computePipelineMatmulTransposeA = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: device.createShaderModule({ code: codeMatmulTransposeA }),
            entryPoint: 'main',
        },
    });

    const computePipelineMatmulTransposeB = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: device.createShaderModule({ code: codeMatmulTransposeB }),
            entryPoint: 'main',
        },
    });

    const computePipelineMatmulTransposeAB = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
            module: device.createShaderModule({ code: codeMatmulTransposeAB }),
            entryPoint: 'main',
        },
    });

    console.log(codeMatmulTransposeB);

    function createBuffer(size, usage) {
        const buffer = Array.from(bufferPool).find(b => b.size === size && b.usage === usage);
        if (buffer) {
            bufferPool.delete(buffer);
            return buffer;
        }
        return device.createBuffer({ size, usage });
    }

    function toGPU(X, usage) {
        const buffer = createBuffer(X.byteLength, usage | GPUBufferUsage.COPY_DST);
        device.queue.writeBuffer(buffer, 0, X);
        return buffer;
    }

    /**
     * @param {Float32Array} A - Matrix A data (shape: [M, K])
     * @param {Float32Array} B - Matrix B data (shape: [K, N])
     * @param {number} aT - transpose flag for A (0 or 1)
     * @param {number} bT - transpose flag for B (0 or 1)
     */
    return async function mm(A, B, aT = 0, bT = 0) {
        const commandEncoder = device.createCommandEncoder();
        const M = aT ? A.shape[1] : A.shape[0];
        const N = bT ? B.shape[0] : B.shape[1];
        const K = aT ? A.shape[0] : A.shape[1];
        if (K !== (bT ? B.shape[1] : B.shape[0])) {
            throw new Error('Matrix dimensions do not match.');
        }

        // Pass transpose flags as uint32 in uniform buffer
        const uniformBuffer = toGPU(
            new Uint32Array([M, N, K, aT, bT]),
            GPUBufferUsage.UNIFORM
        );
        const array_c = toGPU(
            new Float32Array(M * N),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        );
        const array_a = toGPU(A, GPUBufferUsage.STORAGE);
        const array_b = toGPU(B, GPUBufferUsage.STORAGE);

        const bindGroup0 = device.createBindGroup({
            layout: bindGroupLayout0,
            entries: [uniformBuffer, array_c, array_a, array_b].map((buffer, i) => ({
                binding: i,
                resource: { buffer },
            })),
        });

        const passEncoder = commandEncoder.beginComputePass();
        const pipelineMap = [
            [computePipelineMatmul, computePipelineMatmulTransposeB],
            [computePipelineMatmulTransposeA, computePipelineMatmulTransposeAB],
        ];
        const computePipeline = pipelineMap[Number(aT)][Number(bT)];
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup0);
        const workgroupSize = 16;
        const workgroupsX = Math.ceil(N / workgroupSize);
        const workgroupsY = Math.ceil(M / workgroupSize);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        passEncoder.end();

        const readBuffer = createBuffer(array_c.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        commandEncoder.copyBufferToBuffer(array_c, 0, readBuffer, 0, readBuffer.size);

        device.queue.submit([commandEncoder.finish()]);

        await readBuffer.mapAsync(GPUMapMode.READ);
        const C = new FloatMatrix(readBuffer.getMappedRange().slice(0)).reshape([M, N]);
        readBuffer.unmap();

        [readBuffer, uniformBuffer, array_c, array_a, array_b].forEach(buffer => bufferPool.add(buffer));
        return C;
    };
}

export async function GPU() {
    if (!navigator.gpu) {
        throw new Error('GPU not supported. Try using Chrome.');
    }
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const matMul = await createMatMul(device);
    return { matMul };
}
