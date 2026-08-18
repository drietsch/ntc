//! Minimal host tensor (row-major f32) plus `.ntc` dtype decoding.

use half::{bf16, f16};
use ntc_core::NtcError;
use ntc_format::{DType, TensorView};

/// Row-major f32 host tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; shape.iter().product()],
        }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "shape/data mismatch"
        );
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Row `i` of a rank-2 tensor.
    pub fn row(&self, i: usize) -> &[f32] {
        debug_assert_eq!(self.rank(), 2);
        let cols = self.shape[1];
        &self.data[i * cols..(i + 1) * cols]
    }

    /// Decode a `.ntc` tensor view into f32 (LE bytes; F32/F16/BF16 only in V1).
    pub fn from_view(view: &TensorView<'_>) -> Result<Self, NtcError> {
        let shape: Vec<usize> = view.record.shape.iter().map(|&d| d as usize).collect();
        let n: usize = shape.iter().product();
        let bytes = view.bytes;
        let data: Vec<f32> = match view.record.dtype {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            DType::F16 => bytes
                .chunks_exact(2)
                .map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|c| bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
                .collect(),
            other => {
                return Err(NtcError::Format(format!(
                    "tensor `{}`: dtype {} not decodable in V1",
                    view.record.name,
                    other.name()
                )))
            }
        };
        if data.len() != n {
            return Err(NtcError::Format(format!(
                "tensor `{}`: decoded {} elements, shape wants {n}",
                view.record.name,
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }
}
