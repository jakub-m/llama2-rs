/// Offset helps to address a sub-matrix, consistenly for CPU and GPU matrices. The offset is not
/// byte offset but number of elements (so, number of f32 or f16 primitives).
#[derive(Clone, Copy, Debug)]
pub struct Offset {
    pub start: usize,
    pub end: usize,
}

impl Offset {
    pub fn at_elem(i: usize, elem_size: usize) -> Offset {
        let start = i * elem_size;
        let end = start + elem_size;
        Offset { start, end }
    }
}

pub trait SliceFromOffset<T> {
    fn slice_from_offset(&self, offset: Offset) -> &[T];
}

impl<T> SliceFromOffset<T> for &[T] {
    fn slice_from_offset(&self, offset: Offset) -> &[T] {
        &self[offset.start..offset.end]
    }
}
