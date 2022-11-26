/// An edge of the analysed graph, annotated by a fact.
#[cfg_attr(feature = "serialize", derive(Serialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct Edge {
    pub id: usize,
    pub from_node: Option<usize>,
    pub from_out: usize,
    pub to_node: Option<usize>,
    pub fact: TensorFact,
}

/// A graph analyser, along with its current state.
pub struct Analyser {
    // The original output.
    pub output: usize,

    // The graph being analysed.
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    pub prev_edges: Vec<Vec<usize>>,
    pub next_edges: Vec<Vec<usize>>,

    // The execution plan and unused nodes.
    plan: Vec<usize>,

    // The current state of the algorithm.
    pub current_pass: usize,
    pub current_step: usize,
    pub current_direction: bool,
}

impl Analyser {
    /// Constructs an analyser for the given graph.
    ///
    /// The output argument is used to infer an execution plan for the graph.
    /// Changing it won't alter the correctness of the analysis, but it might
    /// take much longer to complete.
    pub fn new(model: Model, output: usize) -> Result<Analyser> {
        let nodes = model.nodes;
        let mut edges = vec![];
        let mut prev_edges = vec![Vec::new(); nodes.len() + 1];
        let mut next_edges = vec![Vec::new(); nodes.len() + 1];

        for node in &nodes {
            for input in &node.inputs {
                let id = edges.len();

                edges.push(Edge {
                    id,
                    from_node: Some(input.0),
                    from_out: input.1.unwrap_or(0),
                    to_node: Some(node.id),
                    fact: TensorFact::new(),
                });

                prev_edges[node.id].push(id);
                next_edges[input.0].push(id);
            }
        }

        // Add a special output edge.
        let special_edge_id = edges.len();
        edges.push(Edge {
            id: special_edge_id,
            from_node: Some(output),
            from_out: 0,
            to_node: None,
            fact: TensorFact::new(),
        });

        next_edges[output].push(special_edge_id);

        // Compute an execution plan for the graph.
        let plan = Plan::for_nodes(&nodes, &[output])?.order;
        let current_pass = 0;
        let current_step = 0;
        let current_direction = true;

        info!("Using execution plan {:?}.", plan);

        Ok(Analyser {
            output,
            nodes,
            edges,
            prev_edges,
            next_edges,
            plan,
            current_pass,
            current_step,
            current_direction,
        })
    }

    /// Adds an user-provided tensor fact to the analyser.
    pub fn hint(&mut self, node: usize, fact: &TensorFact) -> Result<()> {
        if node >= self.next_edges.len() {
            bail!("There is no node with index {:?}.", node);
        }

        for &j in &self.next_edges[node] {
            self.edges[j].fact = unify(fact, &self.edges[j].fact)?;
        }

        Ok(())
    }

    /// Returns a model from the analyser.
    pub fn into_model(self) -> Model {
        let mut nodes_by_name = HashMap::with_capacity(self.nodes.len());
        self.nodes.iter().for_each(|n| {
            nodes_by_name.insert(n.name.clone(), n.id);
        });

        Model {
            nodes: self.nodes,
            nodes_by_name,
        }
    }

    /// Computes a new execution plan for the graph.
    pub fn reset_plan(&mut self) -> Result<()> {
        self.plan = Plan::for_nodes(&self.nodes, &[self.output])?.order;
        Ok(())
    }

    /// Detaches the constant nodes and edges from the given graph.
    pub fn propagate_constants(&mut self) -> Result<()> {
        constants::propagate_constants(self)
    }

    /// Removes the nodes and edges which are not part of the execution plan.
    /// Returns the mapping between the old and new node indexes.
    pub fn prune_unused(&mut self) -> Vec<Option<usize>> {
        let mut node_used = vec![false; self.nodes.len()];
        let mut edge_used = vec![false; self.edges.len()];
        for &i in &self.plan {
            node_used[i] = true;
        }

        // Remove the nodes while keeping track of the new indices.
        let mut deleted = 0;
        let mut node_mapping = vec![None; self.nodes.len()];

        for i in 0..self.nodes.len() {
            if !node_used[i] {
                self.nodes.remove(i - deleted);

                self.prev_edges.remove(i - deleted);
                self.next_edges.remove(i - deleted);
                deleted += 1;
            } else {
                node_mapping[i] = Some(i - deleted);

                self.prev_edges[i - deleted].iter().for_each(|&j| edge_used[j] = true);
                self.next_edges[i - deleted].iter().for_each(|&j| edge_used[j] = true);
            }
        }

        info!("Deleted {:?} unused nodes.", deleted);

        // Update the nodes and edges to use the new indices.
        for node in &mut self.nodes {
            node.id = node_mapping[node.id].unwrap();
            node.inputs.iter_mut().for_each(|i| i.0 = node_mapping[i.0].unwrap());
        }

        for edge in &mut self.edges {
            if let Some(i) = edge.from_node {
                edge.from_node = node_mapping[i];
            }

            if let Some(i) = edge.to_node {
                edge.to_node = node_mapping[i];
            }
        }

        // Remove the edges while keeping track of the new indices.
        let mut deleted = 0;
        let mut edge_mapping = vec![None; self.edges.len()];

        for i in 0..self.edges.len() {
            if !edge_used[i] {
                self.edges.remove(i - deleted);
                deleted += 1;
            } else {
                edge_mapping[i] = Some(i - deleted);
            }
        }

        info!("Deleted {:?} unused edges.", deleted);

        // Update the adjacency lists to use the new indices.
        for i in 0..self.nodes.len() {
            self.prev_edges[i].iter_mut().for_each(|j| *j = edge_mapping[*j].unwrap());
            self.next_edges[i].iter_mut().for_each(|j| *j = edge_mapping[*j].unwrap());
        }

        node_mapping
    }

    /// Runs the entire analysis at once.
    pub fn run(&mut self) -> Result<()> {
        self.current_pass = 0;

        loop {
            if !self.run_two_passes()? {
                return Ok(());
            }
        }
    }

    /// Runs two passes of the analysis.
    pub fn run_two_passes(&mut self) -> Result<bool> {
        let mut changed = false;

        info!(
            "Starting pass [pass={:?}, direction={:?}].",
            self.current_pass, self.current_direction,
        );

        // We first run a forward pass.
        self.current_step = 0;
        for _ in 0..self.plan.len() {
            if self.run_step()? {
                changed = true;
            }
        }

        info!(
            "Starting pass [pass={:?}, direction={:?}].",
            self.current_pass, self.current_direction,
        );

        // We then run a backward pass.
        self.current_step = 0;
        for _ in 0..self.plan.len() {
            if self.run_step()? {
                changed = true;
            }
        }

        Ok(changed)
    }

    /// Runs a single step of the analysis.
    pub fn run_step(&mut self) -> Result<bool> {
        let changed = self.try_step()?;

        // Switch to the next step.
        self.current_step += 1;
        if self.current_step == self.plan.len() {
            self.current_pass += 1;
            self.current_direction = !self.current_direction;
            self.current_step = 0;
        }

        Ok(changed)
    }

    /// Tries to run a single step of the analysis, and returns whether
    /// there was any additional information gained during the step.
    fn try_step(&mut self) -> Result<bool> {
        let node = if self.current_direction {
            &self.nodes[self.plan[self.current_step]]
        } else {
            &self.nodes[self.plan[self.plan.len() - 1 - self.current_step]]
        };

        debug!(
            "Starting step for {} ({}) [pass={:?}, direction={:?}, step={:?}].",
            node.name, node.op_name, self.current_pass, self.current_direction, self.current_step,
        );

        let inputs: Vec<_> = self.prev_edges[node.id]
            .iter()
            .map(|&i| self.edges[i].fact.clone())
            .collect();

        let mut outputs = vec![TensorFact::new()];
        for &i in &self.next_edges[node.id] {
            outputs[0] = unify(&self.edges[i].fact, &outputs[0])?;
        }

        let enriched = node.op
            .enrich(inputs, outputs)
            .map_err(|e| format!("While enriching for {}: {}", node.name, e))?;

        let mut changed = false;

        for (i, &j) in self.prev_edges[node.id].iter().enumerate() {
            let fact = &enriched.0[i];
            let unified = unify(fact, &self.edges[j].fact)
                .map_err(|e| format!(
                    "While unifying inputs of node {:?}: {}",
                    node.name, e
                ))?;

            changed |= unified != self.edges[j].fact;
            self.edges[j].fact = unified;
        }

        for (_, &j) in self.next_edges[node.id].iter().enumerate() {
            if enriched.1.len() != 1 {
                panic!("Analyser only supports nodes with a single output port.");
            }

            let fact = &enriched.1[0];
            let unified = unify(fact, &self.edges[j].fact)
                .map_err(|e| format!(
                    "While unifying outputs of node {:?}: {}",
                    node.name, e
                ))?;

            changed |= unified != self.edges[j].fact;
            self.edges[j].fact = unified;
        }

        Ok(changed)
    }
}
