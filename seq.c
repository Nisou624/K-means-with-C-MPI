/*  This is an implementation of the k-means clustering algorithm (aka Lloyd's algorithm) without MPI. */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>

#define MAX_ITERATIONS 1000

int numOfClusters = 0;
int numOfElements = 0;

/* This function goes through that data points and assigns them to a cluster */
void assign2Cluster(double k_x[], double k_y[], double data_x[], double data_y[], int assign[], int numOfElements, int numOfClusters)
{
    for (int i = 0; i < numOfElements; i++)
    {
        double min_dist = 1e9;
        int k_min_index = 0;

        for (int j = 0; j < numOfClusters; j++)
        {
            double x = data_x[i] - k_x[j];
            double y = data_y[i] - k_y[j];
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
void calcKmeans(double k_means_x[], double k_means_y[], double data_x_points[], double data_y_points[], int k_assignment[], int numOfElements, int numOfClusters)
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
    if (argc != 2)
    {
        printf("Please include an argument after the program name to list how many clusters.\n");
        printf("e.g. To indicate 3 clusters, run: ./kmeans 3\n");
        exit(-1);
    }

    numOfClusters = atoi(argv[1]);
    printf("Ok %d clusters it is.\n", numOfClusters);

    // allocate memory for arrays
    double *k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
    double *k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

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

    // allocate memory for an array of data points
    double *data_x_points = (double *)malloc(sizeof(double) * numOfElements);
    double *data_y_points = (double *)malloc(sizeof(double) * numOfElements);
    int *k_assignment = (int *)malloc(sizeof(int) * numOfElements);

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

    int count = 0;
    while (count < MAX_ITERATIONS)
    {
        // assign the data points to a cluster
        assign2Cluster(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment, numOfElements, numOfClusters);

        // recalculate k means
        calcKmeans(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment, numOfElements, numOfClusters);

        count++;
    }

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

    // deallocate memory and clean up
    free(k_means_x);
    free(k_means_y);
    free(data_x_points);
    free(data_y_points);
    free(k_assignment);

    return 0;
}
