package tudelft.rl.mysolution;

import java.util.ArrayList;

import tudelft.rl.Action;
import tudelft.rl.QLearning;
import tudelft.rl.State;

public class MyQLearning extends QLearning {

	@Override
	public void updateQ(State s, Action a, double r, State s_next, ArrayList<Action> possibleActions, double alfa,
			double gamma) {
		// Pick the actions which will give the maximum value (out of all possible
		// actions from that given state)
		double[] actionValues = getActionValues(s_next, possibleActions);
		double max = Double.MIN_VALUE;
		for (int i = 0; i < actionValues.length; i++) {
			if (actionValues[i] > max) {
				max = actionValues[i];
			}
		}

		// Now update the values in the hashmap using the formula
		/// Q(current state, action i) + alfa(r + gamma * max action value - Q(current
		// state, action i))
		setQ(s, a, getQ(s, a) + alfa * (r + gamma * max - getQ(s, a)));
	}

}
