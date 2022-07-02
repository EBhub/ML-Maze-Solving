package tudelft.rl.mysolution;

import java.io.File;

import tudelft.rl.*;

public class RunMe {

	// Input parameters
	public static double epsilon = 0.5;
	public static double alfa = 0.7;
	public static double gamma = 0.9;

	public static void main(String[] args) {

		// load the maze
		// replace this with the location to your maze on your file system
		Maze maze = new Maze(new File("C:\\CI\\QLearning\\data\\toy_maze.txt"));

		// Set the reward at the bottom right to 10
		maze.setR(maze.getState(9, 9), 10);
		maze.setR(maze.getState(9, 0), 5);

		// create a robot at starting and reset location (0,0) (top left)
		Agent robot = new Agent(0, 0);

		// make a selection object (you need to implement the methods in this class)
		EGreedy selection = new MyEGreedy();

		// make a Qlearning object (you need to implement the methods in this class)
		QLearning learn = new MyQLearning();

		int step = 0; // Keeps track of the amount of steps taken
		boolean stop = false;

		// Keep learning until you decide to stop
		while (!stop) {

			// Chose an action based on the Greedy selection method
			Action chosenAction = selection.getEGreedyAction(robot, maze, learn, epsilon);

			// Perform the chosen action and implement qlearning
			State oldState = robot.getState(maze);
			State newState = robot.doAction(chosenAction, maze);
			learn.updateQ(oldState, chosenAction, maze.getR(robot.getState(maze)), newState,
					maze.getValidActions(robot), alfa, gamma);

			// If the robot reaches the lower right corner, reset it to its starting
			// position
			if (robot.x == 9 && robot.y == 9) {
				robot.reset();
			}

			if (robot.x == 9 && robot.y == 0) {
				robot.reset();
			}

			// Stop a trial after 30000 steps
			if (step < 30000) {
				step++;
				epsilon -= 0.5 / 30000;
			} else {
				stop = true;
			}
		}

	}

}
