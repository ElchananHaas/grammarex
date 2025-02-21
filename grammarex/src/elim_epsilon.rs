use crate::nsms::{Action, CharMatch, Machine, NsmEdgeTransition};

fn build_epsilon_bypasses(machines: &mut Vec<Machine>) {
    let mut made_progress;
    //TODO this can loop infinitely if there is left recursion going on
    //and semantic actions too.
    //For now, just don't allow nullable left recursive loops
    loop {
        made_progress = false;
        for i  in 0..machines.len() {
            let mut visit_set = vec![false; machines[i].edges.len()];
            let mut actions = vec![];
            if traverse_epsilon_edges_rec(machines, &machines[i], &mut visit_set, &mut actions, 0) {
                let actions: Option<Vec<_>> = Some(actions);
                if actions != machines[i].accept_epsilon_actions {
                    made_progress = true;
                }
                machines[i].accept_epsilon_actions = actions;
            }
        }
        if !made_progress {
            break;
        }
    }
}

fn traverse_epsilon_edges_rec(machines: &Vec<Machine>, machine: &Machine, visit_set: &mut Vec<bool>, actions: &mut Vec<Action>, node: usize) -> bool {
    visit_set[node] = true;
    for edge in &machine.edges[node] {
        for action in &edge.actions {
            actions.push(action.clone());
        }
        match &edge.transition {
            NsmEdgeTransition::Move(CharMatch::Epsilon, next_node) => {
                if traverse_epsilon_edges_rec(machines, machine, visit_set, actions, *next_node) {
                    return true;
                }
            }
            NsmEdgeTransition::Return => {
                return true;
            }
            NsmEdgeTransition::Call(call_data) => {
                if let Some(callee_actions) = &machines[call_data.target_machine].accept_epsilon_actions {
                    for action in callee_actions {
                        actions.push(action.clone());
                    }
                    if traverse_epsilon_edges_rec(machines, machine, visit_set, actions, call_data.return_node) {
                        return true;
                    }
                    for _ in callee_actions {
                        actions.pop();
                    }
                }
            }
            _ => {}
        }
        for _ in &edge.actions {
            actions.pop();
        }
    }
    false
}

fn elim_epsilon(machines: &mut Vec<Machine>, idx: usize) {
    let mut new_machine = Machine::new();
    for _ in 0..machines[idx].edges.len() {
        new_machine.create_node();
    }
}