
use crate::{
    nsms::{CallData, CharMatch, LoweringError, MachineBuilder, NsmEdgeData, NsmEdgeTransition},
    types::GrammarEx,
};

fn epsilon_no_actions(target: usize) -> NsmEdgeData {
    NsmEdgeData {
        transition: NsmEdgeTransition::Move(CharMatch::Epsilon, target),
        actions: vec![],
    }
}

// Takes in a machine builder, a start node, a name table from expression names to start nodes and an expresion to lower.
// Returns the end node of the expression lowered as an subgraph with end_node accepting.
fn lower_nsm_rec(
    builder: &mut MachineBuilder,
    start_node: usize,
    expr: GrammarEx,
) -> Result<usize, LoweringError> {
    match expr {
        GrammarEx::Epsilon => Ok(start_node),
        GrammarEx::Char(c) => {
            let end_node = builder.create_node();
            builder.get_node_mut(start_node).push_back(NsmEdgeData{
                transition: NsmEdgeTransition::Move(CharMatch::Char(c), end_node),
                actions: vec![],
            });
            Ok(end_node)
        }
        GrammarEx::CharRangeInclusive(m, n) => {
            let end_node = builder.create_node();
            builder.get_node_mut(start_node).push_back(NsmEdgeData{
                transition: NsmEdgeTransition::Move(CharMatch::RangeInclusive(m, n), end_node),
                actions: vec![],
            });
            Ok(end_node)
        }
        GrammarEx::Seq(exprs) => {
            let mut end_node = start_node;
            for expr in exprs {
                end_node = lower_nsm_rec(builder, end_node, expr)?;
            }
            Ok(end_node)
        }
        GrammarEx::Star(expr) => {
            let end_node = lower_nsm_rec(builder, start_node, *expr)?;
            //For a star operator looping back is always highest priority. Skipping it is lowest priority
            builder.get_node_mut(end_node).push_front(epsilon_no_actions(start_node));
            builder.get_node_mut(start_node).push_back(epsilon_no_actions(end_node));
            Ok(end_node)
        }
        GrammarEx::Plus(expr) => {
            let end_node = lower_nsm_rec(builder, start_node, *expr)?;
            //For a plus operator looping back is always highest priority. It can't be skipped.
            builder.get_node_mut(end_node).push_front(epsilon_no_actions(start_node));
            Ok(end_node)
        }
        GrammarEx::Alt(exprs) => {
            //The code could get away with not creating a node if the
            //alt is non-empty, but that would make it more complicated.
            let end_node = builder.create_node();
            for expr in exprs {
                let end = lower_nsm_rec(builder, start_node, expr)?;
                builder.get_node_mut(end).push_back(epsilon_no_actions(end_node));
            }
            Ok(end_node)
        }
        GrammarEx::Optional(grammar_ex) => {
            let end_node = lower_nsm_rec(builder, start_node, *grammar_ex)?;
            //An option can be skipped, skipping has lowest priority over consuming input
            builder.get_node_mut(start_node).push_back(epsilon_no_actions(end_node));
            Ok(end_node)
        }
        GrammarEx::Assign(var, expr) => {
            let GrammarEx::Var(var) = *var else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let GrammarEx::Var(expr) = *expr else {
                return Err(LoweringError::InvalidVariableAssignment);
            };
            let machine = builder
                .get_name(&expr)?;
            let return_node = builder.create_node();
            let call_data = CallData {
                target_node: 0,
                name: expr,
                return_node,
                target_machine: machine,
            };
            builder.get_node_mut(start_node).push_back(
                NsmEdgeData {
                    transition: NsmEdgeTransition::Call(call_data),
                    actions: vec![],
                },
            );
            Ok(return_node)
        }
        GrammarEx::Var(expr) => {
            let machine = builder
                .get_name(&expr)?;
            let return_node = builder.create_node();
            let call_data = CallData {
                target_node: 0,
                name: expr,
                return_node,
                target_machine: machine,
            };
            builder.get_node_mut(start_node).push_back(
                NsmEdgeData {
                    transition: NsmEdgeTransition::Call(call_data),
                    actions: vec![],
                },
            );
            Ok(return_node)
        }
    }
}