//! PyTorch's CPU random number generator: `torch.manual_seed(seed)`
//! followed by `torch.randn` / `torch.rand` in double precision, bit for bit.
//!
//! BoTorch draws seeded random numbers with torch's default CPU generator
//! (e.g. the base samples of `IIDNormalSampler`: `torch.randn(shape)` under
//! `manual_seed(seed)`). [`TorchGenerator`] reproduces that generator without
//! torch: the 32-bit Mersenne twister of `at::mt19937` (seeded like
//! `init_genrand`), `random64()` from two 32-bit draws (the first one in the
//! high bits), uniform doubles from the low 53 bits, `torch.randn` of 16 or
//! more contiguous doubles by `normal_fill` (all uniforms first, Box–Muller on
//! the pairs `(j, j + 8)` of every block of 16, the last block recomputed
//! with new uniforms if the size is not a multiple of 16) and of fewer values
//! by `at::normal_distribution<double>` (Box–Muller on two uniforms, the
//! second normal cached in the generator until the next draw or seed).

use std::f64::consts::PI;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// PyTorch's default CPU generator (`at::CPUGeneratorImpl`) after
/// `torch.manual_seed(seed)`.
#[derive(Clone, Debug)]
pub struct TorchGenerator {
    state: Box<[u32; N]>,
    left: usize,
    next: usize,
    next_double_normal: Option<f64>,
}

impl TorchGenerator {
    /// The generator of `torch.manual_seed(seed)` (the low 32 bits of the
    /// seed initialize the twister, as in `at::mt19937`).
    pub fn new(seed: u64) -> Self {
        let mut state = Box::new([0u32; N]);
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..N {
            state[j] = 1_812_433_253u32
                .wrapping_mul(state[j - 1] ^ (state[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        Self {
            state,
            left: 1,
            next: 0,
            next_double_normal: None,
        }
    }

    fn next_state(&mut self) {
        let s = &mut self.state;
        self.left = N;
        self.next = 0;
        for i in 0..N - M {
            let y = (s[i] & UPPER_MASK) | (s[i + 1] & LOWER_MASK);
            s[i] = s[i + M] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
        }
        for i in N - M..N - 1 {
            let y = (s[i] & UPPER_MASK) | (s[i + 1] & LOWER_MASK);
            s[i] = s[i + M - N] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
        }
        let y = (s[N - 1] & UPPER_MASK) | (s[0] & LOWER_MASK);
        s[N - 1] = s[M - 1] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
    }

    /// The next 32-bit output of the twister (`random()`).
    pub fn random(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random64()`: two 32-bit outputs, the first in the high bits.
    pub fn random64(&mut self) -> u64 {
        let hi = u64::from(self.random());
        let lo = u64::from(self.random());
        (hi << 32) | lo
    }

    /// A uniform double in `[0, 1)` (`uniform_real_distribution<double>`).
    pub fn uniform(&mut self) -> f64 {
        let v = self.random64() & ((1u64 << 53) - 1);
        v as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// `at::normal_distribution<double>(0, 1)` (one value; the second value
    /// of the Box–Muller pair is cached).
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.next_double_normal.take() {
            return v;
        }
        let u1 = self.uniform();
        let u2 = self.uniform();
        let r = (-2.0 * (-u2).ln_1p()).sqrt();
        let theta = 2.0 * PI * u1;
        self.next_double_normal = Some(r * theta.sin());
        r * theta.cos()
    }

    /// `torch.rand(n, dtype=torch.float64)`.
    pub fn rand(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.uniform()).collect()
    }

    /// A uniform float in `[0, 1)` (`uniform_real_distribution<float>`: the
    /// low 24 bits of one 32-bit output).
    pub fn uniform_f32(&mut self) -> f32 {
        let v = self.random() & ((1u32 << 24) - 1);
        v as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// `torch.rand(n, dtype=torch.float32)`.
    pub fn rand_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.uniform_f32()).collect()
    }

    /// `torch.randn(n, dtype=torch.float64)` (a contiguous tensor).
    pub fn randn(&mut self, n: usize) -> Vec<f64> {
        if n < 16 {
            return (0..n).map(|_| self.normal()).collect();
        }
        let mut data: Vec<f64> = (0..n).map(|_| self.uniform()).collect();
        let mut i = 0;
        while i + 16 <= n {
            normal_fill_16(&mut data[i..i + 16]);
            i += 16;
        }
        if n % 16 != 0 {
            let tail = &mut data[n - 16..];
            for v in tail.iter_mut() {
                *v = self.uniform();
            }
            normal_fill_16(tail);
        }
        data
    }
}

/// Box–Muller on the pairs `(j, j + 8)` of a block of 16 uniforms.
fn normal_fill_16(data: &mut [f64]) {
    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_reference_outputs() {
        // init_genrand(5489): the reference MT19937 outputs.
        let mut g = TorchGenerator::new(5489);
        let expected = [3_499_211_612u32, 581_869_302, 3_890_346_734, 3_586_334_585];
        for e in expected {
            assert_eq!(g.random(), e);
        }
    }

    #[test]
    fn normals_have_the_right_moments() {
        let mut g = TorchGenerator::new(0);
        let z = g.randn(100_001);
        let mean = z.iter().sum::<f64>() / z.len() as f64;
        let var = z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64;
        assert!(mean.abs() < 0.01 && (var - 1.0).abs() < 0.01);
        let u = g.rand(1000);
        assert!(u.iter().all(|&v| (0.0..1.0).contains(&v)));
    }
}
