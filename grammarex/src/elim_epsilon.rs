use std::collections::{HashMap, HashSet};

use crate::{
    lower_grammarex::lower_grammarex,
    nsms::{Action, CallData, CharMatch, Machine, NsmEdgeData, NsmEdgeTransition},
    GrammarEx,
};

fn build_epsilon_bypasses(machines: &mut Vec<Machine>) {
    let mut made_progress;
    //TODO this can loop infinitely if there is left recursion going on
    //and semantic actions too.
    //For now, just don't allow nullable left recursive loops
    loop {
        made_progress = false;
        for i in 0..machines.len() {
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

fn traverse_epsilon_edges_rec(
    machines: &Vec<Machine>,
    machine: &Machine,
    visit_set: &mut Vec<bool>,
    actions: &mut Vec<Action>,
    node: usize,
) -> bool {
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
                if let Some(callee_actions) =
                    &machines[call_data.target_machine].accept_epsilon_actions
                {
                    for action in callee_actions {
                        actions.push(action.clone());
                    }
                    if traverse_epsilon_edges_rec(
                        machines,
                        machine,
                        visit_set,
                        actions,
                        call_data.return_node,
                    ) {
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

fn elim_all_epsilons(machines: &Vec<Machine>) -> Vec<Machine> {
    let mut new_machines = Vec::new();
    for i in 0..machines.len() {
        new_machines.push(elim_epsilons(machines, i));
    }
    new_machines
}

fn elim_epsilons(machines: &Vec<Machine>, idx: usize) -> Machine {
    let mut new_machine = Machine::new();
    for _ in 0..machines[idx].edges.len() {
        new_machine.create_node();
    }
    for i in 0..machines[idx].edges.len() {
        let mut visit_set: HashSet<usize> = HashSet::new();
        let mut actions = Vec::new();
        replace_with_epsilon_closure(
            &machines,
            &machines[idx],
            i,
            i,
            &mut new_machine,
            &mut visit_set,
            &mut actions,
        );
    }
    new_machine.accept_epsilon_actions = machines[idx].accept_epsilon_actions.clone();
    new_machine
}

// Replaces a node's edges with their epsilon closure
fn replace_with_epsilon_closure<'a>(
    machines: &Vec<Machine>,
    current_machine: &Machine,
    start_node: usize,
    current_node: usize,
    new_machine: &mut Machine,
    visit_set: &mut HashSet<usize>,
    actions: &mut Vec<Action>,
) {
    visit_set.insert(current_node);
    for edge in &current_machine.edges[current_node] {
        for action in &edge.actions {
            actions.push(action.clone());
        }
        match &edge.transition {
            NsmEdgeTransition::Call(call_data) => {
                new_machine.edges[start_node].push(NsmEdgeData {
                    transition: NsmEdgeTransition::Call(call_data.clone()),
                    actions: actions.clone(),
                });
                //If the state being called accepts epsilon, bypass it in epsilon elimination
                //If we already processed the node, no need to explore it.
                if !visit_set.contains(&call_data.return_node) {
                    if let Some(bypass_actions) = &machines[0].accept_epsilon_actions {
                        //Add in any actions used in the bypass.
                        for action in bypass_actions {
                            actions.push(action.clone());
                        }
                        replace_with_epsilon_closure(
                            machines,
                            current_machine,
                            start_node,
                            call_data.return_node,
                            new_machine,
                            visit_set,
                            actions,
                        );
                        for _ in bypass_actions {
                            actions.pop();
                        }
                    }
                }
            }
            NsmEdgeTransition::Move(char_match, next_node) => {
                if let CharMatch::Epsilon = &char_match {
                    //If the visit set contains the node, there was already a more preferred way to
                    //get to this node and any nodes reachable from it. So do nothing
                    if !visit_set.contains(&next_node) {
                        //In this case, there is an edge that consumes epsilon. So recursively explore more of the graph.
                        replace_with_epsilon_closure(
                            machines,
                            current_machine,
                            start_node,
                            *next_node,
                            new_machine,
                            visit_set,
                            actions,
                        );
                    }
                } else {
                    new_machine.edges[start_node].push(NsmEdgeData {
                        transition: NsmEdgeTransition::Move(char_match.clone(), *next_node),
                        actions: actions.clone(),
                    });
                }
            }
            NsmEdgeTransition::Return => {
                new_machine.edges[start_node].push(NsmEdgeData {
                    transition: NsmEdgeTransition::Return,
                    actions: actions.clone(),
                });
            }
        }
        for _ in &edge.actions {
            actions.pop();
        }
    }
}

fn deduplicate(machines: &Vec<Machine>) -> Vec<Machine> {
    let mut res = Vec::new();
    let remappings: Vec<_> = machines
        .into_iter()
        .map(|machine| node_dedup_mapping(machine))
        .collect();
    for i in 0..machines.len() {
        res.push(remap(&machines[i], &remappings[i]));
    }
    res
}

fn remap(machine: &Machine, remapping: &(Vec<usize>, usize)) -> Machine {
    let mut res = Machine::new();
    remap_into_machine(&machine, &mut res, remapping);
    res
}
fn remap_into_machine(machine: &Machine, res: &mut Machine, remapping: &(Vec<usize>, usize)) {
    let mut remapped = vec![None; remapping.1];
    for i in 0..machine.edges.len() {
        let mapped_to = remapping.0[i];
        if remapped[mapped_to].is_none() {
            let edges = &machine.edges[i];
            let new_edges: Vec<NsmEdgeData> = edges
                .iter()
                .map(|edge| remap_edge(edge, &remapping))
                .collect();
            remapped[mapped_to] = Some(new_edges);
        }
    }
    res.edges = remapped
        .into_iter()
        .map(|edges| edges.expect("All edges were mapped"))
        .collect();
    res.accept_epsilon_actions = machine.accept_epsilon_actions.clone();
}

fn remap_edge(edge: &NsmEdgeData, remapping: &(Vec<usize>, usize)) -> NsmEdgeData {
    NsmEdgeData {
        transition: match &edge.transition {
            NsmEdgeTransition::Move(char_match, target) => {
                NsmEdgeTransition::Move(char_match.clone(), remapping.0[*target])
            }
            NsmEdgeTransition::Call(call_data) => NsmEdgeTransition::Call(CallData {
                name: call_data.name.clone(),
                target_machine: call_data.target_machine,
                return_node: remapping.0[call_data.return_node],
            }),
            NsmEdgeTransition::Return => NsmEdgeTransition::Return,
        },
        actions: edge.actions.clone(),
    }
}

fn node_dedup_mapping(machine: &Machine) -> (Vec<usize>, usize) {
    let mut count = 0;
    let mut mapping = Vec::new();
    let mut edges_to_node = HashMap::new();
    for edges in &machine.edges {
        if let Some(val) = edges_to_node.get(edges) {
            mapping.push(*val);
        } else {
            edges_to_node.insert(edges.clone(), count);
            mapping.push(count);
            count += 1;
        }
    }
    (mapping, count)
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RetData {
    pub target_machine: usize,
    pub return_node: usize,
}

fn greibachize(machines: &Vec<Machine>, idx: usize) {
    let mut visit_set = vec![false; machines.len()];
    let mut rets: Vec<Vec<RetData>> = vec![Vec::new(); machines.len()];
    let mut starts_and_lens: Vec<Option<(usize, usize)>> = vec![Some((0, 0)); machines.len()];
    let mut res = Machine::new();
    direct_call_graph_visit(
        machines,
        idx,
        &mut visit_set,
        &mut starts_and_lens,
        &mut rets,
        &mut res,
    );
}

fn patch_returns(machine: &mut Machine, starts_and_lens: &Vec<Option<(usize, usize)>>) {
    //Don't remap returns for the first machine.
    for item in &starts_and_lens[1..] {

    }
}
fn direct_call_graph_visit(
    machines: &Vec<Machine>,
    idx: usize,
    visit_set: &mut Vec<bool>,
    starts_and_lens: &mut Vec<Option<(usize, usize)>>,
    rets: &mut Vec<Vec<RetData>>,
    new_machine: &mut Machine,
) {
    if visit_set[idx] {
        return;
    }
    visit_set[idx] = true;
    let remapping_size = machines[idx].edges.len();
    let new_start = new_machine.edges.len();
    starts_and_lens[idx] = Some((new_start, remapping_size));
    let remapping = (
        (new_start..remapping_size).into_iter().collect(),
        remapping_size,
    );
    remap_into_machine(&machines[idx], new_machine, &remapping);
    let machine = &machines[idx];
    for edge in &machine.edges[0] {
        if let NsmEdgeTransition::Call(call_data) = &edge.transition {
            rets[call_data.target_machine].push(RetData {
                target_machine: idx,
                return_node: call_data.return_node,
            });
            direct_call_graph_visit(
                machines,
                call_data.target_machine,
                visit_set,
                starts_and_lens,
                rets,
                new_machine,
            );
        }
    }
}

fn compile(machines: &HashMap<String, GrammarEx>) -> Vec<Machine> {
    let machines = lower_grammarex(&machines).unwrap();
    //Do a deduplication pass before epsilon elimination to reduce the number of nodes.
    //It should give the same result as performing epsilon elimination before deduplicating,
    //but deduplication is relatively cheap.
    let mut machines = deduplicate(&machines);
    build_epsilon_bypasses(&mut machines);
    let machines = elim_all_epsilons(&machines);
    //Epsilon elimination creates duplicate nodes, remove them.
    let machines = deduplicate(&machines);
    machines
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::parse_grammarex;

    use super::*;

    #[test]
    fn test_lower_machine() {
        let expr = parse_grammarex(&mut "[cba]").unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr);
        let machines = compile(&machines);
        dbg!(machines);
    }

    #[test]
    fn test_lower_machine_multi() {
        let expr_one = parse_grammarex(&mut r#" abc = second "#).unwrap();
        let expr_two = parse_grammarex(&mut r#"[abc]"#).unwrap();
        let start = "start".to_string();
        let second = "second".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        machines.insert(second.clone(), expr_two);
        let machines = compile(&machines);
        dbg!(machines);
    }

    #[test]
    fn test_lower_rec() {
        let expr_one = parse_grammarex(&mut r#" "a" | \( start \) "#).unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        let machines = compile(&machines);
        dbg!(machines);
    }

    #[test]
    fn test_lower_rec_two() {
        let expr_one = parse_grammarex(&mut r#"   "a" "b"  | "c" start  "#).unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        let machines = compile(&machines);
        dbg!(machines);
    }

    #[test]
    fn test_lower_call_epsilon() {
        let expr_one = parse_grammarex(&mut r#" abc = second "#).unwrap();
        let expr_two = parse_grammarex(&mut r#" "x"? "#).unwrap();
        let start = "start".to_string();
        let second = "second".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        machines.insert(second.clone(), expr_two);
        let machines = compile(&machines);
        dbg!(machines);
    }

    #[test]
    fn test_lower_call_epsilon_two() {
        let expr_one = parse_grammarex(&mut r#" abc = second "x" "#).unwrap();
        let expr_two = parse_grammarex(&mut r#" "x"? "#).unwrap();
        let start = "start".to_string();
        let second = "second".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        machines.insert(second.clone(), expr_two);
        let machines = compile(&machines);
        dbg!(machines);
    }

    #[test]
    fn test_lower_left_recursive_loop() {
        let expr_one = parse_grammarex(&mut r#"  start? "a"   "#).unwrap();
        let start = "start".to_string();
        let mut machines = HashMap::new();
        machines.insert(start.clone(), expr_one);
        let machines = compile(&machines);
        dbg!(machines);
    }
}
