import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.regex.Pattern;

/**
 * Finds shortest path between two points in a maze according to a specific path specification.
 */
public class AntColonyOptimization {
	
	private int antsPerGen;
    private int generations;
    private double Q;
    private double evaporation;
    private Maze maze;

    /**
     * Constructs a new optimization object using ants.
     * @param maze the maze .
     * @param antsPerGen the amount of ants per generation.
     * @param generations the amount of generations.
     * @param Q normalization factor for the amount of dropped pheromone
     * @param evaporation the evaporation factor.
     */
    public AntColonyOptimization(Maze maze, int antsPerGen, int generations, double Q, double evaporation) {
        this.maze = maze;
        this.antsPerGen = antsPerGen;
        this.generations = generations;
        this.Q = Q;
        this.evaporation = evaporation;
    }

    /**
     * Loop that starts the shortest path process
     * @param spec Spefication of the route we wish to optimize
     * @return ACO optimized route
     */
    public Route findShortestRoute(PathSpecification spec) {
        maze.reset();
        Route shortest=null;
        for(int i=0;i<generations;i++){
            Ant[] ants=new Ant[antsPerGen];
            ArrayList<Route> routes=new ArrayList<Route>(antsPerGen);
            for(int j=0;j<antsPerGen;j++){
                ants[j]=new Ant(maze,spec);
                routes.add(ants[j].findRoute());
                if(i==0&&j==0){
                    shortest=routes.get(0);
                }
                else if(routes.get(routes.size()-1).shorterThan(shortest)){
                    shortest=routes.get(routes.size()-1);
                }
            }
            System.out.println(shortest.size());
            maze.evaporate(evaporation);
            maze.addPheromoneRoutes(routes,Q);
        }
        return shortest;
    }

    /**
     * Driver function
     */
    public static void main(String[] args) throws FileNotFoundException {
    	//parameters
    	int gen = 20;
        int noGen = 8;
        double Q = 1500;
        double evap = 0.5;

        //construct the optimization objects
        Maze maze = Maze.createMaze("D:\\Users\\Ruben\\IdeaProjects\\group-13\\ACO\\data\\hard maze.txt");
        PathSpecification spec = PathSpecification.readCoordinates("D:\\Users\\Ruben\\IdeaProjects\\group-13\\ACO\\data\\hard coordinates.txt");
        AntColonyOptimization aco = new AntColonyOptimization(maze, gen, noGen, Q, evap);

        //save starting time
        long startTime = System.currentTimeMillis();
        
        //run optimization
        Route shortestRoute = aco.findShortestRoute(spec);
        
        //print time taken
        System.out.println("Time taken: " + ((System.currentTimeMillis() - startTime) / 1000.0));
        
        //save solution
        shortestRoute.writeToFile("./data/hard_solution.txt");
        
        //print route size
        System.out.println("Route size: " + shortestRoute.size());
    }
}
