use std::{
    fs::File,
    io::{BufReader, Result},
    path::Path,
};

use byteorder::{LittleEndian, ReadBytesExt};

trait ReadVec<T> {
    fn read_element(reader: &mut BufReader<File>) -> Result<T>;
}

impl ReadVec<f32> for f32 {
    fn read_element(reader: &mut BufReader<File>) -> Result<f32> {
        reader.read_f32::<LittleEndian>()
    }
}

impl ReadVec<i32> for i32 {
    fn read_element(reader: &mut BufReader<File>) -> Result<i32> {
        reader.read_i32::<LittleEndian>()
    }
}

fn read_vecs<T, P>(file_path: P) -> Result<Vec<Vec<T>>>
where
    P: AsRef<Path>,
    T: ReadVec<T>,
{
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    while let Ok(dim) = reader.read_i32::<LittleEndian>() {
        let mut vec = Vec::with_capacity(dim as usize);
        for _ in 0..dim {
            let val = T::read_element(&mut reader)?;
            vec.push(val);
        }
        vectors.push(vec);
    }

    Ok(vectors)
}

pub fn get_data() -> Result<(Vec<Vec<f32>>, Vec<Vec<i32>>, Vec<Vec<f32>>)> {
    let current_dir = std::env::current_dir()?;
    let siftsmall_dir = current_dir.join("src/data/examples/siftsmall");

    let vectors = read_vecs::<f32, _>(siftsmall_dir.join("siftsmall_base.fvecs"))?;
    let query_vectors = read_vecs::<f32, _>(siftsmall_dir.join("siftsmall_query.fvecs"))?;
    let groundtruth = read_vecs::<i32, _>(siftsmall_dir.join("siftsmall_groundtruth.ivecs"))?;

    return Ok((query_vectors.clone(), groundtruth.clone(), vectors.clone()));
}
