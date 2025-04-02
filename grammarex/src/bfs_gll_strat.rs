use std::{cell::UnsafeCell, num::NonZeroUsize};

use crate::nsms::{Machine, NsmEdgeTransition};

pub struct ItemHeader {
    //The machine type
    machine: usize,
    //The current state within the machine
    state: usize,
    //An index to a bitset for tracking the active states
    bitset_idx: usize,
}

impl ItemHeader {
    fn new(machine: usize, state: usize, bitset_idx: usize) -> Self {
        Self {
            machine,
            state,
            bitset_idx,
        }
    }

    pub fn set_state(&mut self, state: usize) {
        self.state = state;
    }

    pub fn set_bitset_idx(&mut self, bitset_idx: usize) {
        self.bitset_idx = bitset_idx;
    }

    pub fn machine(&self) -> usize {
        self.machine
    }

    pub fn state(&self) -> usize {
        self.state
    }

    pub fn bitset_idx(&self) -> usize {
        self.bitset_idx
    }
}

struct WaveFront {
    items: Vec<UnsafeCell<Option<ItemHeader>>>,
    to_clear: UnsafeCell<Vec<usize>>,
}

impl WaveFront {
    fn new(num_entries: usize) -> Self {
        Self {
            items: (0..num_entries).map(|_| UnsafeCell::new(None)).collect(),
            to_clear: UnsafeCell::new(Vec::new()),
        }
    }

    fn get_or_init(&self, machine: usize, mut f: impl FnMut() -> ItemHeader) -> &ItemHeader {
        let to_clear;
        let item;
        unsafe {
            to_clear = &mut *self.to_clear.get();
            item = &mut *self.items[machine].get()
        }
        //Safety: If there is an immutable reference to a value this method
        //will return another immutable reference and not alter the value.
        //It will only mutate the value if it is None, in which case there
        //are no references to it.
        item.get_or_insert_with(|| {
            to_clear.push(machine);
            f()
        })
    }

    //Safety: There must be no outstanding references to items in the container.
    unsafe fn clear(&mut self) {
        let to_clear;
        unsafe {
            to_clear = &mut *self.to_clear.get();
        }
        for machine in &*to_clear {
            self.items[*machine] = UnsafeCell::new(None);
        }
        to_clear.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MachineRef(usize);

impl MachineRef {
    const INDEX_BITS: usize = 40;
    pub fn new(machine: usize, idx: usize) -> Self {
        //This is doing some bit packing so debug check numbers
        //are in the expected ranges and check that usize is 64 bits.
        assert!(size_of::<usize>() == 8);
        debug_assert!(machine < (1 << (64 - Self::INDEX_BITS)));
        debug_assert!(idx < (1 << Self::INDEX_BITS));
        MachineRef((machine << Self::INDEX_BITS) | idx)
    }

    pub fn index(self) -> usize {
        self.0 & ((1 << Self::INDEX_BITS) - 1)
    }

    pub fn machine(self) -> usize {
        self.0 >> Self::INDEX_BITS
    }
}

#[derive(Clone, Debug)]
struct MachineState {
    //Current state
    cur_state: usize,
    //This is the location of a structure for tracking which states are active and its parent states.
    active_states_idx: usize,
    //This will get a field for [Local Variable State]
}

#[derive(Debug)]
struct MachineArena {
    states: Vec<MachineState>,
}

#[derive(Debug)]
//This stores the states of a given machine.
struct PerMachineSet {
    //TODO - replace with bitset or something even more efficient.
    //The current active states. This is reset before each round of GLL.
    states: Vec<bool>,
    //All nodes within a per machine set share the same parents. This
    //can proabaly be replaced with a linked list structure.
    parents: Vec<MachineRef>,
}

#[derive(Debug)]
struct Gss {
    frozen: FrozenStates,
    counted: CountedStates,
}

impl Gss {
    fn new(machines: &Vec<Machine>) -> Self {
        let frozen = machines
            .into_iter()
            .map(|_machine| MachineArena { states: vec![] })
            .collect();
        let counted = machines
            .into_iter()
            .map(|machine| PerMachineSetArena {
                num_states: machine.edges.len(),
                sets: vec![],
            })
            .collect();

        Gss {
            frozen: FrozenStates { frozen },
            counted: CountedStates { counted },
        }
    }
}
#[derive(Debug)]
struct CountedStates {
    counted: Vec<PerMachineSetArena>,
}

impl CountedStates {
    fn create(&mut self, arena: usize) -> usize {
        let num_states = self.counted[arena].num_states;
        self.counted[arena].sets.push(PerMachineSet {
            states: vec![false; num_states],
            parents: Vec::new(),
        });
        self.counted[arena].sets.len() - 1
    }

    fn get(&self, arena: usize, idx: usize) -> &PerMachineSet {
        &self.counted[arena].sets[idx]
    }

    fn get_mut(&mut self, arena: usize, idx: usize) -> &mut PerMachineSet {
        &mut self.counted[arena].sets[idx]
    }
}

#[derive(Debug)]
struct PerMachineSetArena {
    num_states: usize,
    sets: Vec<PerMachineSet>,
}

#[derive(Debug)]
struct FrozenStates {
    frozen: Vec<MachineArena>,
}

impl FrozenStates {
    fn create_state(&mut self, state: MachineState, arena: usize) -> MachineRef {
        self.frozen[arena].states.push(state);
        MachineRef::new(arena, self.frozen[arena].states.len() - 1)
    }

    fn get(&self, machine_ref: MachineRef) -> &MachineState {
        &self.frozen[machine_ref.machine()].states[machine_ref.index()]
    }
}

pub fn run(machines: &Vec<Machine>, input: &str) -> Vec<ItemHeader> {
    let mut gss = Gss::new(machines);
    let set = gss.counted.create(0);
    let init_state = ItemHeader::new(0, 0, set);
    let mut machine_refs = vec![init_state];
    let mut wavefront = WaveFront::new(machines.len());
    for (count, char) in input.chars().enumerate() {
        //TODO reset bitsets!!! This is a bug RN.
        let mut new_states = vec![];
        for r in &machine_refs {
            advance_machine(
                &mut gss,
                machines,
                r,
                &mut new_states,
                &wavefront,
                count,
                char,
            );
        }
        //SAFETY: There are no outstanding references to items in the wavefront.
        unsafe {
            wavefront.clear();
        }
        machine_refs = new_states;
    }
    machine_refs
}

fn advance_machine(
    gss: &mut Gss,
    machines: &Vec<Machine>,
    current_state: &ItemHeader,
    new_states: &mut Vec<ItemHeader>,
    call_wavefront: &WaveFront,
    count: usize,
    c: char,
) {
    for edge in &machines[current_state.machine()].edges[current_state.state()] {
        match &edge.transition {
            NsmEdgeTransition::Move(char_match, target) => {
                if char_match.matches(c) {
                    if !gss
                        .counted
                        .get(current_state.machine(), current_state.bitset_idx())
                        .states[*target]
                    {
                        gss.counted
                            .get_mut(current_state.machine(), current_state.bitset_idx())
                            .states[*target] = true;
                        new_states.push(ItemHeader::new(
                            current_state.machine(),
                            *target,
                            current_state.bitset_idx(),
                        ));
                    }
                }
            }
            NsmEdgeTransition::Call(call_data) => {
                let return_state = gss.frozen.create_state(
                    MachineState {
                        cur_state: call_data.return_node,
                        active_states_idx: current_state.bitset_idx(),
                    },
                    current_state.machine(),
                );
                let new_active_state = call_wavefront.get_or_init(call_data.target_machine, || {
                    let set = gss.counted.create(call_data.target_machine);
                    ItemHeader::new(call_data.target_machine, 0, set)
                });
                //Advance the new machine on the given char.
                //Prior transformations guarantee this won't lead to infinite recursion.
                advance_machine(
                    gss,
                    machines,
                    new_active_state,
                    new_states,
                    call_wavefront,
                    count,
                    c,
                );
                gss.counted
                    .get_mut(new_active_state.machine(), new_active_state.bitset_idx())
                    .parents
                    .push(return_state);
            }
            NsmEdgeTransition::Return => {
                for i in 0..gss
                    .counted
                    .get(current_state.machine(), current_state.bitset_idx())
                    .parents
                    .len()
                {
                    let frozen_ref = gss
                        .counted
                        .get(current_state.machine(), current_state.bitset_idx())
                        .parents[i];
                    let state = gss.frozen.get(frozen_ref).clone();
                    let current = ItemHeader::new(
                        frozen_ref.machine(),
                        state.cur_state,
                        state.active_states_idx,
                    );
                    advance_machine(
                        gss,
                        machines,
                        &current,
                        new_states,
                        call_wavefront,
                        count,
                        c,
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{elim_epsilon::compile, parse_grammarex};

    use super::*;

    #[test]
    fn test_a() {
        let expr = parse_grammarex(&mut "[cba]").unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr);
        let machines: Vec<Machine> = compile(&machines).unwrap();
        let res = run(&machines, "a");
        assert!(res.len() == 1);
        let res = run(&machines, "c");
        assert!(res.len() == 1);
        let res = run(&machines, "d");
        assert!(res.len() == 0);
        let res = run(&machines, "aa");
        assert!(res.len() == 0);
    }

    #[test]
    fn test_rec() {
        let expr_one = parse_grammarex(&mut r#" "a" | \( start \) "#).unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        let machines = compile(&machines).unwrap();
        let res = run(&machines, "(a)");
        assert!(res.len() == 1);
        let res = run(&machines, "(a");
        assert!(res.len() == 1);
        let res = run(&machines, "(ab");
        assert!(res.len() == 0);
    }
}
