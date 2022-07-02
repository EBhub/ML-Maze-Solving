import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * TSP problem solver using genetic algorithms.
 */
public class GeneticAlgorithm {

    private int generations;
    private int popSize;

    /**
     * Constructs a new 'genetic algorithm' object.
     * @param generations the amount of generations.
     * @param popSize the population size.
     */
    public GeneticAlgorithm(int generations, int popSize) {
        this.generations = generations;
        this.popSize = popSize;
    }


    /**
     * Knuth-Yates shuffle, reordering a array randomly
     * @param chromosome array to shuffle.
     */
    private void shuffle(int[] chromosome) {
        int n = chromosome.length;
        for (int i = 0; i < n; i++) {
            int r = i + (int) (Math.random() * (n - i));
            int swap = chromosome[r];
            chromosome[r] = chromosome[i];
            chromosome[i] = swap;
        }
    }

    /**
     * This method should solve the TSP. 
     * @param pd the TSP data.
     * @return the optimized product sequence.
     */
    public int[] solveTSP(TSPData pd) {
        double pcrossover=0.7;
        double pmutate=0.1;

        ArrayList<int[]> oldpopulation=buildPopulation();
        for(int i=1;i<generations;i++) {
            int[] distances = getPathLength(oldpopulation, pd);
            double[] probabilities = getProbabilities(distances);
            int minDistance = 0;
            for(int j = 0; j < distances.length; j++) {
                if(distances[minDistance] > distances[j]) {
                    minDistance = j;
                }
            }
            System.out.println(Arrays.toString(oldpopulation.get(minDistance)));
            ArrayList<int[]> newpopulation = makeNewPopulation(oldpopulation, probabilities);
            newpopulation = crossOver(newpopulation, pcrossover);
            newpopulation = mutate(newpopulation, pmutate);
            oldpopulation = newpopulation;

            int sum=0;
            for(int j=0;j<distances.length;j++){
                sum+=distances[j];
            }
            System.out.println(sum/distances.length);


        }

        int[] distances = getPathLength(oldpopulation, pd);
        int min=Integer.MAX_VALUE;
        int[] best=oldpopulation.get(0);
        for(int i=0;i<distances.length;i++){
            if(distances[i]<min){
                min=distances[i];
                best=oldpopulation.get(i);
            }
        }
        return best;
    }

    public ArrayList<int[]> buildPopulation(){
        int[] chromosome=new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

        ArrayList<int[]> population=new ArrayList<int[]>(popSize);
        for(int i=0;i<popSize;i++){
            shuffle(chromosome);
            int[] newchromo=Arrays.copyOf(chromosome,chromosome.length);
            population.add(0,newchromo);
        }
        return population;
    }
    public int[] getPathLength(ArrayList<int[]> population, TSPData pd){
        int[] lengths=new int[popSize];
        for(int i=0;i<popSize;i++){
            int[] route=population.get(i);
            int length=pd.getStartDistances()[route[0]];
            for(int j=0;j<route.length-1;j++){
                length+=pd.getDistances()[route[j]][route[j+1]];
            }
            lengths[i]=length;
        }
        return lengths;
    }
    public double[] getProbabilities(int[] distances){
        double[] probs=new double[distances.length];
        for(int i=0;i<distances.length;i++){
            probs[i]=1.0/distances[i];
        }
        return probs;
    }
    public ArrayList<int[]> makeNewPopulation(ArrayList<int[]> oldpopulation, double[] probs){
        ArrayList<int[]> newpopulation=new ArrayList<int[]>(popSize);
        Random rand=new Random();
        double totalprobs=0;
        for(int i=0;i<probs.length;i++){
            totalprobs+=probs[i];
        }
        for(int i=0;i<popSize;i++){
            double r=rand.nextDouble()*totalprobs;
            double sum=0;
            int choice=0;
            for(int j=0;j<probs.length;j++){
                sum+=probs[j];
                if(r<sum){
                    choice=j;
                    break;
                }
            }
            newpopulation.add(oldpopulation.get(choice));
        }
        return newpopulation;
    }
    public ArrayList<int[]> crossOver(ArrayList<int[]> oldpopulation, double p){
        ArrayList<int[]> newpopulation=new ArrayList<int[]>();
        Random r=new Random();
        while(oldpopulation.size()>1){
            //Chose two random chromosomes from old generation
            int[] parent1=oldpopulation.remove(r.nextInt(oldpopulation.size()));
            int[] parent2=oldpopulation.remove(r.nextInt(oldpopulation.size()));
            int[] child1, child2;

            //Random chance that crossover happens
            if(r.nextDouble()<p){
                //We perform double crossover, based on two randomly chosen spots
                int posL =r.nextInt(parent1.length);
                int posR = r.nextInt(parent1.length);
                if(posL > posR) {
                    int tmp = posL;
                    posL = posR;
                    posR = tmp;
                }

                child1=new int[parent1.length];
                child2=new int[parent2.length];
                for(int i=posL;i<=posR;i++){
                    child1[i]=parent2[i];
                    child2[i]=parent1[i];
                }
                int index1 = 0;
                int index2 = 0;
                if(posL == 0) {
                    index1 = posR + 1;
                    index2 = posR + 1;
                }
                for(int i=0;i<parent1.length;i++){
                    if(index1 < parent1.length){
                        //Check if the gene is already in the chromosome
                        boolean isContained=false;
                        for(int j=0;j<index1;j++){
                            if(child1[j]==parent1[i]){
                                isContained=true;
                            }
                        }
                        for(int j=posL;j<=posR;j++){
                            if(child1[j]==parent1[i]){
                                isContained=true;
                            }
                        }
                        //If it is not, add it and increase index
                        if(!isContained){
                            child1[index1]=parent1[i];
                            if(index1 + 1 == posL) {
                                index1 = posR + 1;
                            } else {
                                index1++;
                            }
                        }
                    }
                    if(index2 < parent2.length){
                        boolean isContained=false;
                        for(int j=0;j<index2;j++){
                            if(child2[j]==parent2[i]){
                                isContained=true;
                            }
                        }
                        for(int j=posL;j<=posR;j++){
                            if(child2[j]==parent2[i]){
                                isContained=true;
                            }
                        }
                        if(!isContained){
                            child2[index2]=parent2[i];
                            if(index2 + 1 == posL) {
                                index2 = posR + 1;
                            } else {
                                index2++;
                            }
                        }
                    }

                }
            }
            else{
                child1=parent1;
                child2=parent2;
            }
            newpopulation.add(child1);
            newpopulation.add(child2);
        }
        if(oldpopulation.size()==1){
            newpopulation.add(oldpopulation.remove(0));
        }
        return newpopulation;
    }
    public ArrayList<int[]> mutate(ArrayList<int[]> oldpopulation, double p){
        ArrayList<int[]> newpopulation=new ArrayList<int[]>();
        for(int i=0;i<popSize;i++){
            newpopulation.add(Arrays.copyOf(oldpopulation.get(i),oldpopulation.get(i).length));
        }
        Random r=new Random();
        for(int i=0;i<popSize;i++){
            if(r.nextDouble()<p){
                int pos1=r.nextInt(oldpopulation.get(i).length);
                int pos2=r.nextInt(oldpopulation.get(i).length);
                while(pos1==pos2){
                    pos2=r.nextInt(oldpopulation.get(i).length);
                }
                newpopulation.get(i)[pos1]=oldpopulation.get(i)[pos2];
                newpopulation.get(i)[pos2]=oldpopulation.get(i)[pos1];
            }
        }
        return newpopulation;
    }

    
    public static void main(String[] args) throws IOException, ClassNotFoundException {
    	//parameters
    	int populationSize = 50;
        int generations = 1000;
        String persistFile = "./data/productMatrixDist.txt";
        
        //setup optimization
        TSPData tspData = TSPData.readFromFile(persistFile);
        GeneticAlgorithm ga = new GeneticAlgorithm(generations, populationSize);
        
        //run optimization and write to file
        int[] solution = ga.solveTSP(tspData);
        tspData.writeActionFile(solution, "./data/TSP solution.txt");
    }
}
