use crate::nsms::{Machine, NsmEdgeTransition};

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

#[derive(Clone)]
struct MachineState {
    //Current state
    cur_state: usize,
    //This is the location of a structure for tracking which states are active and its parent states.
    active_states_idx: usize,
    //This will get a field for [Local Variable State]
}

struct MachineArena {
    states: Vec<MachineState>,
}

//This stores the states of a given machine.
struct PerMachineSet {
    //TODO - replace with bitset or something even more efficient.
    //The current active states. This is reset before each round of GLL.
    states: Vec<bool>,
    //All nodes within a per machine set share the same parents. This
    //can proabaly be replaced with a linked list structure.
    parents: Vec<MachineRef>,
}

struct Gss {
    active: ActiveStates,
    frozen: FrozenStates,
    counted: CountedStates,
}

impl Gss {
    fn new(machines: &Vec<Machine>) -> Self {
        let active = machines
            .into_iter()
            .map(|_machine| MachineArena { states: vec![] })
            .collect();
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
            active: ActiveStates { active },
            frozen: FrozenStates { frozen },
            counted: CountedStates { arenas: counted },
        }
    }
}
struct CountedStates {
    arenas: Vec<PerMachineSetArena>,
}

impl CountedStates {
    fn create(&mut self, arena: usize) -> usize {
        let num_states = self.arenas[arena].num_states;
        self.arenas[arena].sets.push(PerMachineSet {
            states: vec![false; num_states],
            parents: Vec::new(),
        });
        self.arenas[arena].sets.len() - 1
    }

    fn get(&self, arena: usize, idx: usize) -> &PerMachineSet {
        &self.arenas[arena].sets[idx]
    }

    fn get_mut(&mut self, arena: usize, idx: usize) -> &mut PerMachineSet {
        &mut self.arenas[arena].sets[idx]
    }
}

struct PerMachineSetArena {
    num_states: usize,
    sets: Vec<PerMachineSet>,
}

struct ActiveStates {
    active: Vec<MachineArena>,
}
impl ActiveStates {
    fn get(&self, machine_ref: MachineRef) -> &MachineState {
        &self.active[machine_ref.machine()].states[machine_ref.index()]
    }

    fn get_mut(&mut self, machine_ref: MachineRef) -> &mut MachineState {
        &mut self.active[machine_ref.machine()].states[machine_ref.index()]
    }

    fn create_state(&mut self, state: MachineState, arena: usize) -> MachineRef {
        self.active[arena].states.push(state);
        MachineRef::new(arena, self.active[arena].states.len() - 1)
    }
}

struct FrozenStates {
    frozen: Vec<MachineArena>,
}

impl FrozenStates {
    fn create_state(&mut self, state: MachineState, arena: usize) -> MachineRef {
        self.frozen[arena].states.push(state);
        MachineRef::new(arena, self.frozen[arena].states.len() - 1)
    }
}

fn run(machines: &Vec<Machine>, input: &str) -> Vec<MachineRef> {
    let mut gss = Gss::new(machines);
    let set = gss.counted.create(0);
    let init_state = gss.active.create_state(
        MachineState {
            cur_state: 0,
            active_states_idx: set,
        },
        0,
    );
    let mut machine_refs = vec![init_state];
    for char in input.chars() {
        let mut new_states = vec![];
        let mut new_machine_active_state_idx = vec![None; machines.len()];
        for r in &machine_refs {
            advance_machine(
                &mut gss,
                machines,
                *r,
                &mut new_states,
                &mut new_machine_active_state_idx,
                char,
            );
        }
        machine_refs = new_states;
    }
    dbg!(&machine_refs);
    machine_refs
}

fn advance_machine(
    gss: &mut Gss,
    machines: &Vec<Machine>,
    machine_ref: MachineRef,
    new_states: &mut Vec<MachineRef>,
    new_machine_active_state_idx: &mut Vec<Option<MachineRef>>,
    c: char,
) {
    let state = gss.active.get(machine_ref).clone();
    for edge in &machines[machine_ref.machine()].edges[state.cur_state] {
        match &edge.transition {
            NsmEdgeTransition::Move(char_match, target) => {
                if char_match.matches(c) {
                    if !gss
                        .counted
                        .get(machine_ref.machine(), state.active_states_idx)
                        .states[*target]
                    {
                        gss.counted
                            .get_mut(machine_ref.machine(), state.active_states_idx)
                            .states[*target] = true;
                        let new_state = MachineState {
                            cur_state: *target,
                            active_states_idx: state.active_states_idx,
                        };
                        let new_ref = gss.active.create_state(new_state, machine_ref.machine());
                        new_states.push(new_ref);
                    }
                }
            }
            NsmEdgeTransition::Call(call_data) => {
                let return_state = gss.frozen.create_state(
                    MachineState {
                        cur_state: call_data.return_node,
                        active_states_idx: state.active_states_idx,
                    },
                    machine_ref.machine(),
                );
                let new_active_state = new_machine_active_state_idx[call_data.target_machine]
                    .unwrap_or_else(|| {
                        let set = gss.counted.create(call_data.target_machine);
                        gss.active.create_state(
                            MachineState {
                                cur_state: 0,
                                active_states_idx: set,
                            },
                            call_data.target_machine,
                        )
                    });
                //Advance the new machine on the given char.
                //Prior transformations guarantee this won't lead to infinite recursion.
                advance_machine(
                    gss,
                    machines,
                    new_active_state,
                    new_states,
                    new_machine_active_state_idx,
                    c,
                );
                let set_idx = gss.active.get(new_active_state).active_states_idx;
                gss.counted
                    .get_mut(new_active_state.machine(), set_idx)
                    .parents
                    .push(return_state);
            }
            NsmEdgeTransition::Return => {
                for i in 0..gss
                    .counted
                    .get(machine_ref.machine(), state.active_states_idx)
                    .parents
                    .len()
                {
                    let frozen_ref = gss
                        .counted
                        .get(machine_ref.machine(), state.active_states_idx)
                        .parents[i];
                    let state = gss.active.get(machine_ref).clone();
                    let new_active = gss.active.create_state(state, frozen_ref.machine());
                    advance_machine(
                        gss,
                        machines,
                        new_active,
                        new_states,
                        new_machine_active_state_idx,
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
        let machines = compile(&machines).unwrap();
        dbg!(&machines);
        let res = run(&machines, "a");
        assert!(res.len() == 1);
        let res = run(&machines, "c");
        assert!(res.len() == 1);
        let res = run(&machines, "d");
        assert!(res.len() == 0);
        let res = run(&machines, "aa");
        assert!(res.len() == 0);
    }
}
