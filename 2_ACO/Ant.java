import java.util.Random;
import java.util.ArrayList;

/**
 * Class that represents the ants functionality.
 */
public class Ant {

    private Maze maze;
    private Coordinate start;
    private Coordinate end;
    private Coordinate currentPosition;
    private ArrayList<Coordinate> visited;
    private static Random rand;
    private Direction lastDirection;

    /**
     * Constructor for ant taking a Maze and PathSpecification.
     * 
     * @param maze Maze the ant will be running in.
     * @param spec The path specification consisting of a start coordinate and an
     *             end coordinate.
     */
    public Ant(Maze maze, PathSpecification spec) {
        this.maze = maze;
        this.start = spec.getStart();
        this.end = spec.getEnd();
        this.currentPosition = start;
        if (rand == null) {
            rand = new Random();
        }
        this.visited = new ArrayList<>();
        visited.add(start);
    }

    /**
     * Method that performs a single run through the maze by the ant.
     * 
     * @return The route the ant found through the maze.
     */
    public Route findRoute() {
        Route route = new Route(start);
        while (!currentPosition.equals(end)) {
            SurroundingPheromone sp = maze.getSurroundingPheromone(currentPosition);
            double[] probs = new double[4];
            probs[0] = sp.get(Direction.North) / sp.getTotalSurroundingPheromone();
            probs[1] = sp.get(Direction.East) / sp.getTotalSurroundingPheromone();
            probs[2] = sp.get(Direction.South) / sp.getTotalSurroundingPheromone();
            probs[3] = sp.get(Direction.West) / sp.getTotalSurroundingPheromone();

            double choice = rand.nextDouble();

            if (choice < probs[0]) {
                currentPosition = currentPosition.add(Direction.North);
                lastDirection = Direction.North;
                if (!removeCycle(currentPosition, route)) {
                    route.add(Direction.North);
                }
            } else if (choice < probs[0] + probs[1]) {
                currentPosition = currentPosition.add(Direction.East);
                lastDirection = Direction.East;
                if (!removeCycle(currentPosition, route)) {
                    route.add(Direction.East);
                }
            } else if (choice < probs[0] + probs[1] + probs[2]) {
                currentPosition = currentPosition.add(Direction.South);
                lastDirection = Direction.South;
                if (!removeCycle(currentPosition, route)) {
                    route.add(Direction.South);
                }
            } else {
                currentPosition = currentPosition.add(Direction.West);
                lastDirection = Direction.West;
                if (!removeCycle(currentPosition, route)) {
                    route.add(Direction.West);
                }
            }
        }
        return route;
    }

    public boolean removeCycle(Coordinate pos, Route r) {
        for (int i = 0; i < visited.size(); i++) {
            if (visited.get(i).equals(pos)) {
                while (visited.size() > i + 1) {
                    visited.remove(visited.size() - 1);
                    r.removeLast();
                }
                return true;
            }
        }
        visited.add(pos);
        return false;
    }
}
