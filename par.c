#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int numOfClusters = 0;
int numOfElements = 0;
int num_of_processes = 0;

/* Cette fonction parcourt les points de données et les assigne à un cluster */
void assignerAuxClusters(double k_x[], double k_y[], double recv_x[], double recv_y[], int assign[], int start, int end, int numOfClusters)
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

            // nouvelle distance minimale trouvée
            if (temp_dist < min_dist)
            {
                min_dist = temp_dist;
                k_min_index = j;
            }
        }

        // mettre à jour l'assignation du cluster de ce point de données
        assign[i] = k_min_index;
    }
}

/* Recalculer les k-means de chaque cluster car chaque point de données peut avoir
   été réassigné à un nouveau cluster pour chaque itération de l'algorithme */
void calculerKmeans(double k_means_x[], double k_means_y[], double data_x_points[], double data_y_points[], int k_assignment[], int numOfElements, int numOfClusters)
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
    // initialiser l'environnement MPI
    MPI_Init(NULL, NULL);

    // obtenir le nombre de processus
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // obtenir le rang
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // buffers d'envoi
    double *k_means_x = NULL;  // valeurs x correspondantes des k-means
    double *k_means_y = NULL;  // valeurs y correspondantes des k-means
    int *k_assignment = NULL;  // chaque point de données est assigné à un cluster
    double *data_x_points = NULL;
    double *data_y_points = NULL;

    // buffer de réception
    double *recv_x = NULL;
    double *recv_y = NULL;
    int *recv_assign = NULL;

    int max_iterations = 0;

    if (world_rank == 0)
    {
        if (argc != 4)
        {
            printf("Veuillez inclure des arguments après le nom du programme pour indiquer combien de processus, le nombre de clusters et le nombre d'itérations.\n");
            printf("par exemple, pour indiquer 4 processus, 3 clusters et 1000 itérations, exécutez : mpirun -n 4 ./kmeans 4 3 1000\n");
            exit(-1);
        }

        num_of_processes = atoi(argv[1]);
        numOfClusters = atoi(argv[2]);
        max_iterations = atoi(argv[3]);

        // diffuser le nombre de clusters à tous les nœuds
        MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allouer de la mémoire pour les tableaux
        k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
        k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

        if (k_means_x == NULL || k_means_y == NULL)
        {
            perror("malloc");
            exit(-1);
        }

        FILE *fp = fopen("input.txt", "r");

        if (!fp)
        {
            perror("fopen");
            exit(-1);
        }

        // compter le nombre de lignes pour connaître le nombre d'éléments
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

        // diffuser le nombre d'éléments à tous les nœuds
        MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allouer de la mémoire pour un tableau de points de données
        data_x_points = (double *)malloc(sizeof(double) * numOfElements);
        data_y_points = (double *)malloc(sizeof(double) * numOfElements);
        k_assignment = (int *)malloc(sizeof(int) * numOfElements);

        if (data_x_points == NULL || data_y_points == NULL || k_assignment == NULL)
        {
            perror("malloc");
            exit(-1);
        }

        // réinitialiser le pointeur de fichier à l'origine du fichier
        fseek(fp, 0, SEEK_SET);

        // maintenant lire les points et remplir les tableaux
        int i = 0;

        double point_x = 0, point_y = 0;

        while (fscanf(fp, "%lf %lf", &point_x, &point_y) != EOF)
        {
            data_x_points[i] = point_x;
            data_y_points[i] = point_y;

            // assigner les k-means initiaux à zéro
            k_assignment[i] = 0;
            i++;
        }

        // fermer le pointeur de fichier
        fclose(fp);

        // sélectionner aléatoirement les k-means initiaux
        time_t t;
        srand((unsigned)time(&t));
        int random;
        for (int i = 0; i < numOfClusters; i++)
        {
            random = rand() % numOfElements;
            k_means_x[i] = data_x_points[random];
            k_means_y[i] = data_y_points[random];
        }
    }
    else
    { // Je suis un nœud de travail

        num_of_processes = atoi(argv[1]);
        numOfClusters = atoi(argv[2]);
        max_iterations = atoi(argv[3]);

        // recevoir la diffusion du nombre de clusters
        MPI_Bcast(&numOfClusters, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // recevoir la diffusion du nombre d'éléments
        MPI_Bcast(&numOfElements, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allouer de la mémoire pour les tableaux
        k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
        k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

        if (k_means_x == NULL || k_means_y == NULL)
        {
            perror("malloc");
            exit(-1);
        }
    }

    // allouer de la mémoire pour les buffers de réception
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

    // Distribuer le travail entre tous les nœuds. Les points de données eux-mêmes resteront constants et
    // ne changeront pas pendant la durée de l'algorithme.
    int *sendcounts = (int *)malloc(sizeof(int) * world_size);
    int *displs = (int *)malloc(sizeof(int) * world_size);

    for (int i = 0; i < world_size; i++)
    {
        sendcounts[i] = elements_per_process + (i < remainder ? 1 : 0);
        displs[i] = i * elements_per_process + (i < remainder ? i : remainder);
    }

    MPI_Scatterv(data_x_points, sendcounts, displs, MPI_DOUBLE, recv_x, local_numOfElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(data_y_points, sendcounts, displs, MPI_DOUBLE, recv_y, local_numOfElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double start_time = MPI_Wtime();
    int count = 0;
    while (count < max_iterations)
    {
        // diffuser les tableaux de k-means
        MPI_Bcast(k_means_x, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(k_means_y, numOfClusters, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // assigner les points de données à un cluster
        assignerAuxClusters(k_means_x, k_means_y, recv_x, recv_y, recv_assign, 0, local_numOfElements, numOfClusters);

        // rassembler les assignations de clusters
        MPI_Gatherv(recv_assign, local_numOfElements, MPI_INT, k_assignment, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

        // laisser le processus racine recalculer les k-means
        if (world_rank == 0)
        {
            calculerKmeans(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment, numOfElements, numOfClusters);
        }

        count++;
    }
    double end_time = MPI_Wtime();

    if (world_rank == 0)
    {
        printf("Execution time: %f seconds\n", end_time - start_time);
        printf("--------------------------------------------------\n");
        printf("FINAL RESULTS:\n");
        for (int i = 0; i < numOfClusters; i++)
        {
            printf("Cluster #%d : (%f, %f)\n", i, k_means_x[i], k_means_y[i]);
        }
        printf("--------------------------------------------------\n");

        // Écrire les résultats finaux dans un fichier de sortie
        FILE *output_fp = fopen("results/output_par.txt", "w");
        if (!output_fp)
        {
            perror("fopen");
            exit(-1);
        }

        for (int i = 0; i < numOfElements; i++)
        {
            fprintf(output_fp, "%f %f %d\n", data_x_points[i], data_y_points[i], k_assignment[i]);
        }

        // Écrire les centroïdes dans le fichier de sortie
        for (int i = 0; i < numOfClusters; i++)
        {
            fprintf(output_fp, "C %f %f %d\n", k_means_x[i], k_means_y[i], i);
        }

        fclose(output_fp);
    }

    // désallouer la mémoire et nettoyer
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
