function initMatrix(v) {
    const array = empty([v.length, v[0].length]);
    const [m, n] = array.shape;

    for ( let m_ = m; m_--; ) {
        for ( let n_ = n; n_--; ) {
            array[m_ * n + n_] = v[m_][n_];
        }
    }

    return array;
}

async function GPU() {
    const adapter = await navigator.gpu.requestAdapter()
    const device = await adapter.requestDevice()
    const source = await fetch( 'matmul.wgsl' ).then( ( response ) => response.text() );

    function createMatMul( device ) {
        const bindGroupLayout0 = device.createBindGroupLayout({
            entries: [
                // Uniforms for Meta
                { 
                    binding: 0, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'uniform' }
                },
                // Storage buffer for array_c
                { 
                    binding: 1, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'storage' }
                }
            ]
        });

        const bindGroupLayout1 = device.createBindGroupLayout({
            entries: [
                // Storage buffer for array_a
                { 
                    binding: 0, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'read-only-storage' }
                },
                // Storage buffer for array_b
                { 
                    binding: 1, 
                    visibility: GPUShaderStage.COMPUTE, 
                    buffer: { type: 'read-only-storage' }
                },
            ],
        } );

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [ bindGroupLayout0, bindGroupLayout1 ],
        });

        const computePipeline = device.createComputePipeline({
            layout: pipelineLayout,
            compute: {
              module: device.createShaderModule({
                code: source,
              }),
              entryPoint: "main"
            }
        });

        function toGPU( X ) {
            const buffer = device.createBuffer({
                size: X.byteLength,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            device.queue.writeBuffer( buffer, 0, X );
            return buffer;
        }

        return async function mm(A, B) {
            const commandEncoder = device.createCommandEncoder();
            const M = A.shape[0];
            const N = B.shape[1];
            const K = A.shape[1];
            const uniformData = new Uint32Array([M, N, K]);
            const uniformBuffer = device.createBuffer({
                size: uniformData.byteLength,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true,
            });
            const mappedRange = uniformBuffer.getMappedRange();
            new Uint32Array(mappedRange).set(uniformData);
            uniformBuffer.unmap();

            const array_c = toGPU( new Float32Array(M*N) );

            const bindGroup0 = device.createBindGroup({
                layout: bindGroupLayout0,  // This should be the layout for group(0)
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: uniformBuffer
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: array_c
                        }
                    }
                ]
            });
            const bindGroup1 = device.createBindGroup({
                layout: bindGroupLayout1,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: toGPU( A )
                        }
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: toGPU( B )
                        }
                    },
                ]
            });

            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(computePipeline);
            passEncoder.setBindGroup(0, bindGroup0);
            passEncoder.setBindGroup(1, bindGroup1);
            const workgroupsX = Math.ceil(N / 16); // 16 being the x-dimension of the workgroup size
            const workgroupsY = Math.ceil(M / 16); // 16 being the y-dimension of the workgroup size
            passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
            passEncoder.end();
            const readBuffer = device.createBuffer({
                size: array_c.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });
            commandEncoder.copyBufferToBuffer(array_c, 0, readBuffer, 0, readBuffer.size);
            device.queue.submit([commandEncoder.finish()]);
            await readBuffer.mapAsync(GPUMapMode.READ);
            const C = new Float32Array(readBuffer.getMappedRange());
            C.shape = [M, N];
            return C;
        }
    }

    return {
        matMul: createMatMul( device ),
    };
}

( async () => {
    window.gpu = await GPU()

    console.log( await window.gpu.matMul(
        initMatrix( [
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
        ] ),
        initMatrix( [
            [ 2, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 2, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 2, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 2, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 2, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 2, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 2, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 2 ],
        ] ),
    ) )

    console.log( matMul(
        initMatrix( [
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
            [ 1, 2, 3, 4, 5, 6, 7, 8 ],
        ] ),
        initMatrix( [
            [ 2, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 2, 0, 0, 0, 0, 0, 0 ],
            [ 0, 0, 2, 0, 0, 0, 0, 0 ],
            [ 0, 0, 0, 2, 0, 0, 0, 0 ],
            [ 0, 0, 0, 0, 2, 0, 0, 0 ],
            [ 0, 0, 0, 0, 0, 2, 0, 0 ],
            [ 0, 0, 0, 0, 0, 0, 2, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 2 ],
        ] ),
    ) )
} )()
