
import { Value, FloatMatrix } from './3-0-makemore-MLP-utils.js';

Value.addOperation('batchNorm', (A, gain, bias) => {
    const n = A.shape.at(-1);
    const restDims = A.shape.slice(0, -1);
    const m = restDims.reduce((a, b) => a * b, 1);
    const bnraw = new FloatMatrix(A);
    const bnmean = createFloatMatrix( [n] );
    const bnvar = createFloatMatrix( [n] );
    const bnvarinv = createFloatMatrix( [n] );

    for (let n_ = n; n_--;) {
        let sum = 0;
        for (let m_ = m; m_--;) {
            sum += A[m_ * n + n_];
        }
        bnmean[n_] = sum / m;
    }

    for (let n_ = n; n_--;) {
        let variance = 0;
        for (let m_ = m; m_--;) {
            variance += (A[m_ * n + n_] - bnmean[n_]) ** 2;
        }
        // Apply Bessel's correction here
        bnvar[n_] = variance / (m - 0);
        bnvarinv[n_] = 1 / Math.sqrt(bnvar[n_] + 1e-5);
    }

    const bnout = createFloatMatrix( A.shape );

    for (let m_ = m; m_--;) {
        for (let n_ = n; n_--;) {
            const i = m_ * n + n_;
            bnraw[i] = (A[i] - bnmean[n_]) * bnvarinv[n_];
            bnout[i] = gain[n_] * bnraw[i] + bias[n_];
        }
    }

    return [
        bnout,
        (grad) => {
            // bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0) - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
            const dA = new FloatMatrix(A);
            const outGradSum = createFloatMatrix( [n] );
            const outGradXbnrawSum = createFloatMatrix( [n] );
            const dGain = createFloatMatrix( gain.shape );
            const dBias = createFloatMatrix( bias.shape );

            // Calculate sums along the batch dimension (m)
            for (let n_ = n; n_--;) {
                for (let m_ = m; m_--;) {
                    const i = m_ * n + n_;
                    outGradSum[n_] += grad[i];
                    outGradXbnrawSum[n_] += grad[i] * bnraw[i];
                    dGain[n_] += grad[i] * bnraw[i];
                    dBias[n_] += grad[i];
                }
            }

            // Calculate the gradient
            for (let m_ = m; m_--;) {
                for (let n_ = n; n_--;) {
                    const i = m_ * n + n_;
                    dA[i] = gain[n_] * bnvarinv[n_] / m * (
                        m * grad[i] - 
                        outGradSum[n_] - 
                        m / (m - 0) * bnraw[i] * outGradXbnrawSum[n_]
                    );
                }
            }

            return [dA, dGain, dBias];
        },
    ];
});
