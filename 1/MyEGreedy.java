package tudelft.rl.mysolution;

import tudelft.rl.Action;
import tudelft.rl.Agent;
import tudelft.rl.EGreedy;
import tudelft.rl.Maze;
import tudelft.rl.QLearning;

import java.util.*;

public class MyEGreedy extends EGreedy {

	@Override
	/**
	 * Takes a random action between 0 and the max amount of possible valid actions
	 * from the agent r in maze m
	 *
	 * @param Agent r
	 * @param Maze  m
	 * @return Action res
	 */
	public Action getRandomAction(Agent r, Maze m) {
		// select an action at random in State s
		Random random = new Random();
		int rand = random.nextInt(m.getValidActions(r).size());
		Action res = m.getValidActions(r).get(rand);

		return res;
	}

	/**
	 * Returns the action that will give the best Q-value according to known values
	 *
	 * @param Agent     r
	 * @param Maze      m
	 * @param Qlearning q
	 * @return Action res
	 */
	@Override
	public Action getBestAction(Agent r, Maze m, QLearning q) {
		ArrayList<Action> actions = m.getValidActions(r);

		// Shuffle all valid actions, such that not the same action is chosen in the
		// case of equal values
		Collections.shuffle(actions);

		double[] actionValues = q.getActionValues(r.getState(m), actions);

		// Pick the actions which will give the maximum value (out of all possible
		// actions from that given state)
		int j = 0;
		double max = Double.MIN_VALUE;
		for (int i = 0; i < actionValues.length; i++) {
			if (actionValues[i] > max) {
				j = i;
				max = actionValues[i];
			}
		}

		return actions.get(j);
	}

	/**
	 * Returns either a random unexplored action, or follows the best Q-value
	 * according to know values. Decision between exploration and following known
	 * paths is based on the possibility exploration: epsilon.
	 *
	 * @param Agent     r
	 * @param Maze      m
	 * @param Qlearning q
	 * @param double    epsilon
	 * @return Action
	 */
	@Override
	public Action getEGreedyAction(Agent r, Maze m, QLearning q, double epsilon) {
		// select between random or best action selection based on epsilon.
		Random random = new Random();

		if (random.nextDouble() < epsilon) {
			return getRandomAction(r, m);
		} else {
			return getBestAction(r, m, q);
		}
	}

}
