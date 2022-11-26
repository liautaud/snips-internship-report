/// Evaluates one step of the operation on the given input tensors.
fn step(
    &self,
    mut inputs: Vec<(Option<usize>, Option<TensorView>)>,
    buffer: &mut Box<OpBuffer>,
) -> Result<Option<Vec<TensorView>>> {
    // We only support the VALID padding strategy for now, with the
    // streaming dimension being either the width or the height.

    // The idea is that, regardless of the strides, we need at least
    // as many chunks in the buffer as the size of the filter in the
    // streaming dimension to compute our first output chunk. Then,
    // we pop the min(buffer_size, k) first chunks from the buffer,
    // ignore the next max(k - buffer_size, 0) chunks, and wait for
    // the k following chunks to compte one output chunk, with k the
    // strides in the streaming dimension.

    let (mut data, mut filter) = args_2!(inputs);

    if filter.0.is_some() || filter.1.is_none() {
        bail!("Filter input should not be streamed.");
    }

    if data.0.is_none()  {
        bail!("Data input should be streamed.");
    }

    // Maybe there is no incoming chunk.
    if data.1.is_none() {
        return Ok(None);
    }

    // Maybe the data is streamed along the batch dimension.
    let dim = data.0.unwrap();
    if dim == 0 {
        let result = self.eval(vec![
            data.1.take().unwrap(),
            filter.1.take().unwrap()
        ])?;

        return Ok(Some(result))
    }

    if dim < 1 || dim > 2 {
        bail!("Conv2D only supports batch, width and height streaming.");
    }

    let data = data.1.take().unwrap().into_tensor();
    let data = into_4d(T::tensor_into_array(data)?)?;
    let data_size = data.shape()[dim];
    debug_assert!(data_size == 1);

    let filter = filter.1.take().unwrap();
    let filter = T::tensor_to_view(&*filter)?;
    let filter_size = filter.shape()[dim - 1];

    // Generates an empty 4-dimensional array of the right shape.
    let empty_array = || {
        match dim {
            1 => Array::zeros((data.shape()[0], 0, data.shape()[2], data.shape()[3])),
            2 => Array::zeros((data.shape()[0], data.shape()[1], 0, data.shape()[3])),
            _ => panic!()
        }
    };

    let buffer = buffer.downcast_mut::<Buffer<T>>()
        .ok_or("The buffer can't be downcasted to Buffer<T>.")?;

    if buffer.prev.is_none() {
        buffer.prev = Some(empty_array());
    }

    let skip = &mut buffer.skip;
    let prev = buffer.prev.as_mut().unwrap();

    if *skip > 0 {
        *skip -= 1;
        return Ok(None)
    }

    let mut next = stack(Axis(dim), &[prev.view(), data.view()])?;
    let next_size = next.shape()[dim];

    // Maybe we don't have enough chunks to compute the convolution yet.
    if next_size < filter_size {
        *skip = 0;
        *prev = next;
        return Ok(None)
    }

    // Otherwise we compute the convolution using the non-streaming implementation.
    let result = self.convolve(&next, filter, dim != 1, dim != 2)?.into_dyn();
    let stride = [self.0.v_stride, self.0.h_stride][dim - 1];

    if stride > next_size {
        // Maybe we must pop more chunks from the buffer than it currently contains.
        *skip = stride - next_size;
        *prev = empty_array();
    } else {
        // Otherwise we pop the right number of chunks to prepare the next iteration.
        next.slice_axis_inplace(Axis(dim), Slice::from(stride..));
        *skip = 0;
        *prev = next;
    }

    Ok(Some(vec![T::array_into_tensor(result).into()]))
}