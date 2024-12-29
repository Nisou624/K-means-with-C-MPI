/*  This is an implementation of the k-means clustering algorithm (aka Lloyd's algorithm) using MPI (message passing interface). */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>
#include <mpi.h>

#define MAX_ITERATIONS 1000

int numOfClusters = 0;
int numOfElements = 0;
int num_of_processes = 0;

/* This function goes through that data points and assigns them to a cluster */
void assign2Cluster(double k_x[], double k_y[], double recv_x[], double recv_y[], int assign[], int start, int end)
{
    for (int i = start; i < end; i++)
    {
        double min_dist = 1e9;
        int k_min_index = 0;

        for (int j = 0; j < numOfClusters; j++)
        {
            double x = recv_x[i] - k_x[j];
            double y = recv_y[i] - k_y[j];
            double temp_dist = sqrt((x * x) + (y * y));

            // new minimum distance found
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                k_min_index = j;
            }
        }

        // update the cluster assignment of this data point
        assign[i] = k_min_index;
    }
}

/* Recalculate k-means of each cluster because each data point may have
   been reassigned to a new cluster for each iteration of the algorithm */
void calcKmeans(double k_means_x[], double k_means_y[], double data_x_points[], double data_y_points[], int k_assignment[])
{
    double total_x = 0;
    double total_y = 0;
    int numOfpoints = 0;

    for (int i = 0; i < numOfClusters; i++)
    {
        total_x = 0;
        total_y = 0;
        numOfpoints = 0;

        for (int j = 0; j < numOfElements; j++)
        {
            if (k_assignment[j] == i)
            {
                total_x += data_x_points[j];
                total_y += data_y_points[j];
                numOfpoints++;
            }
        }

        if (numOfpoints != 0)
        {
            k_means_x[i] = total_x / numOfpoints;
            k_means_y[i] = total_y / numOfpoints;
        }
    }
}

int main(int argc, char *argv[])
{
    // initialize the MPI environment
    MPI_Init(NULL, NULL);

    // get number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // get rank
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // send buffers
    double *k_means_x = NULL;  // k means corresponding x values
    double *k_means_y = NULL;  // k means corresponding y values
    int *k_assignment = NULL;  // each data point is assigned to a cluster
    double *data_x_points = NULL;
    double *data_y_points = NULL;

    // receive buffer
    double *recv_x = NULL;
    double *recv_y = NULL;
    int *recv_assign = NULL;

    if (world_rank == 0)
    {
        if (argc != 2)
        {
            printf("Please include an argument after the program name to list how many processes.\n");
            printf("e.g. To indicate 4 processes, run: mpirun -n 4 ./kmeans 4\n");
            exit(-1);
        }

        num_of_processes = atoi(argv[1]);

        char buffer[2];
        printf("How many clusters would you like to analyze for? ");
        scanf("%s", buffer);
        printf("\n");

        numOfClusters = atoi(buffer);
        printf("Ok %d clusters it is.\n", numOfClusters);

        // broadcast the number of clusters to all nodes
        MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allocate memory for arrays
        k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
        k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

        if (k_means_x == NULL || k_means_y == NULL)
        {
            perror("malloc");
            exit(-1);
        }

        printf("Reading input data from file...\n\n");

        FILE *fp = fopen("input.txt", "r");

        if (!fp)
        {
            perror("fopen");
            exit(-1);
        }

        // count number of lines to find out how many elements
        int c = 0;
        numOfElements = 0;
        while (!feof(fp))
        {
            c = fgetc(fp);
            if (c == '\n')
            {
                numOfElements++;
            }
        }

        printf("There are a total number of %d elements in the file.\n", numOfElements);

        // broadcast the number of elements to all nodes
        MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allocate memory for an array of data points
        data_x_points = (double *)malloc(sizeof(double) * numOfElements);
        data_y_points = (double *)malloc(sizeof(double) * numOfElements);
        k_assignment = (int *)malloc(sizeof(int) * numOfElements);

        if (data_x_points == NULL || data_y_points == NULL || k_assignment == NULL)
        {
            perror("malloc");
            exit(-1);
        }

        // reset file pointer to origin of file
        fseek(fp, 0, SEEK_SET);

        // now read in points and fill the arrays
        int i = 0;

        double point_x = 0, point_y = 0;

        while (fscanf(fp, "%lf %lf", &point_x, &point_y) != EOF)
        {
            data_x_points[i] = point_x;
            data_y_points[i] = point_y;

            // assign the initial k means to zero
            k_assignment[i] = 0;
            i++;
        }

        // close file pointer
        fclose(fp);

        // randomly select initial k-means
        time_t t;
        srand((unsigned)time(&t));
        int random;
        for (int i = 0; i < numOfClusters; i++)
        {
            random = rand() % numOfElements;
            k_means_x[i] = data_x_points[random];
            k_means_y[i] = data_y_points[random];
        }

        printf("Running k-means algorithm for %d iterations...\n\n", MAX_ITERATIONS);
        for (int i = 0; i < numOfClusters; i++)
        {
            printf("Initial K-means: (%f, %f)\n", k_means_x[i], k_means_y[i]);
        }
    }
    else
    { // I am a worker node

        num_of_processes = atoi(argv[1]);

        // receive broadcast of number of clusters
        MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // receive broadcast of number of elements
        MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allocate memory for arrays
        k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
        k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

        if (k_means_x == NULL || k_means_y == NULL)
        {
            perror("malloc");
            exit(-1);
        }
    }

    // allocate memory for receive buffers
    int elements_per_process = numOfElements / num_of_processes;
    int remainder = numOfElements % num_of_processes;
    int local_numOfElements = elements_per_process + (world_rank < remainder ? 1 : 0);

    recv_x = (double *)malloc(sizeof(double) * local_numOfElements);
    recv_y = (double *)malloc(sizeof(double) * local_numOfElements);
    recv_assign = (int *)malloc(sizeof(int) * local_numOfElements);

    if (recv_x == NULL || recv_y == NULL || recv_assign == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    // Distribute the work among all nodes. The data points itself will stay constant and
    // not change for the duration of the algorithm.
    int *sendcounts = (int *)malloc(sizeof(int) * world_size);
    int *displs = (int *)malloc(sizeof(int) * world_size);

    for (int i = 0; i < world_size; i++)
    {
        sendcounts[i] = elements_per_process + (i < remainder ? 1 : 0);
        displs[i] = i * elements_per_process + (i < remainder ? i : remainder);
    }

    MPI_Scatterv(data_x_points, sendcounts, displs, MPI_DOUBLE, recv_x, local_numOfElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(data_y_points, sendcounts, displs, MPI_DOUBLE, recv_y, local_numOfElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int count = 0;
    while (count < MAX_ITERATIONS)
    {
        // broadcast k-means arrays
        MPI_Bcast(k_means_x, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_y, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // assign the data points to a cluster
        assign2Cluster(k_means_x, k_means_y, recv_x, recv_y, recv_assign, 0, local_numOfElements);

        // gather back k-cluster assignments
        MPI_Gatherv(recv_assign, local_numOfElements, MPI_INT, k_assignment, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        // let the root process recalculate k means
        if (world_rank == 0)
        {
            calcKmeans(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment);
        }

        count++;
    }

    if (world_rank == 0)
    {
        printf("--------------------------------------------------\n");
        printf("FINAL RESULTS:\n");
        for (int i = 0; i < numOfClusters; i++)
        {
            printf("Cluster #%d: (%f, %f)\n", i, k_means_x[i], k_means_y[i]);
        }
        printf("--------------------------------------------------\n");

        // Write the final results to an output file
        FILE *output_fp = fopen("output.txt", "w");
        if (!output_fp)
        {
            perror("fopen");
            exit(-1);
        }

        for (int i = 0; i < numOfElements; i++)
        {
            fprintf(output_fp, "%f %f %d\n", data_x_points[i], data_y_points[i], k_assignment[i]);
        }

        // Write the centroids to the output file
        for (int i = 0; i < numOfClusters; i++)
        {
            fprintf(output_fp, "C %f %f %d\n", k_means_x[i], k_means_y[i], i);
        }

        fclose(output_fp);
    }

    // deallocate memory and clean up
    free(k_means_x);
    free(k_means_y);
    free(data_x_points);
    free(data_y_points);
    free(k_assignment);
    free(recv_x);
    free(recv_y);
    free(recv_assign);
    free(sendcounts);
    free(displs);

    MPI_Finalize();
}
