const bufferPool = new Set();

async function createMatMul( device ) {
    const code = await fetch( 'matmul.wgsl' ).then( ( response ) => response.text() );
    const bindGroupLayout0 = device.createBindGroupLayout({
        // Buffers for Meta, array_c, array_a, array_b
        entries: [ 'uniform', 'storage', 'read-only-storage', 'read-only-storage' ].map( ( type, i ) => ( {
            binding: i,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type },
        } ) )
    });

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [ bindGroupLayout0 ],
    });

    const computePipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: {
          module: device.createShaderModule({ code }),
          entryPoint: "main"
        }
    });

    function createBuffer(size, usage) {
        const buffer = Array.from(bufferPool).find((buffer) => {
            return buffer.size === size && buffer.usage === usage;
        });
        if (buffer) {
            bufferPool.delete(buffer);
            return buffer;
        }
        return device.createBuffer({
            size,
            usage,
        });
    }

    function toGPU( X, usage ) {
        const buffer = createBuffer( X.byteLength, usage | GPUBufferUsage.COPY_DST );
        device.queue.writeBuffer( buffer, 0, X );
        return buffer;
    }

    return async function mm(A, B) {
        const commandEncoder = device.createCommandEncoder();
        const M = A.shape[0];
        const N = B.shape[1];
        const K = A.shape[1];
        const uniformBuffer = toGPU( new Uint32Array([M, N, K]), GPUBufferUsage.UNIFORM );
        const array_c = toGPU( new Float32Array(M*N), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC );
        const array_a = toGPU( A, GPUBufferUsage.STORAGE );
        const array_b = toGPU( B, GPUBufferUsage.STORAGE );
        const bindGroup0 = device.createBindGroup({
            layout: bindGroupLayout0,
            entries: [ uniformBuffer, array_c, array_a, array_b ].map( ( buffer, i ) => ( {
                binding: i,
                resource: { buffer },
            } ) )
        });

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup0);
        const workgroupSize = 16;
        // Ceil because we want to make sure we cover the entire array
        // including trailing rows/cols (not just the multiple of 16)
        const workgroupsX = Math.ceil(N / workgroupSize);
        const workgroupsY = Math.ceil(M / workgroupSize);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        passEncoder.end();
        const readBuffer = createBuffer( array_c.size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST );
        commandEncoder.copyBufferToBuffer(array_c, 0, readBuffer, 0, readBuffer.size);
        device.queue.submit([commandEncoder.finish()]);
        await readBuffer.mapAsync(GPUMapMode.READ);
        const C = new Float32Array(readBuffer.getMappedRange()).slice(0);
        C.shape = [M, N];
        readBuffer.unmap();
        [ readBuffer, uniformBuffer, array_c, array_a, array_b ].forEach( buffer => bufferPool.add( buffer ) );
        return C;
    }
}

export async function GPU() {
    if ( !navigator.gpu ) {
        return;
    }
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const matMul = await createMatMul( device );
    return { matMul };
}
