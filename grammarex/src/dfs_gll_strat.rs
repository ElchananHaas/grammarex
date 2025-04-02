use bitset::Bitsets;

mod bitset {
    pub struct Bitsets {
        //TODO replace this with something much more efficient
        data: Vec<Vec<bool>>,
    }

    impl Bitsets {
        pub fn new() -> Self {
            Self { data: vec![] }
        }

        //Set a bit. This expand the bitsets size if needed.
        pub fn set(&mut self, index: usize, item: usize, bitwidth: usize) {
            if index >= self.data.len() {
                self.data.push(vec![false; bitwidth]);
            }
            self.data[index][item] = true;
        }

        //Bitwidth will be used in the more efficient representation.
        pub fn get(&mut self, index: usize, item: usize, _bitwidth: usize) -> bool {
            if index <= self.data.len() {
                self.data[index][item]
            } else {
                false
            }
        }
    }
}

struct MachineStarts {
    starts: Vec<MachineStart>,
}

struct MachineStart {
    //This will contain the data the Machine produces.
}

fn run_machine(char_iter: &mut (impl Iterator<Item = char> + Clone)) {
    let mut index = 0;
    while let Some(c) = char_iter.next() {
        index += 1;
    }
}
