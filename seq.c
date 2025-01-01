#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>

int numOfClusters = 0;
int iterations = 0;
int numOfElements = 0;

/* Cette fonction parcourt les points de données et les assigne à un cluster */
void assignerAuxClusters(double k_x[], double k_y[], double data_x[], double data_y[], int assign[], int numOfElements, int numOfClusters)
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
    if (argc != 3)
    {
        printf("Veuillez inclure un argument après le nom du programme pour indiquer combien de clusters et d'itérations.\n");
        printf("ex. Pour indiquer 5 itérations, exécutez ./kmeans 3 5\n");
        exit(-1);
    }

    numOfClusters = atoi(argv[1]);
    printf("D'accord, ce sera %d clusters.\n", numOfClusters);
    iterations = atoi(argv[2]);
    printf("D'accord, ce sera %d itérations.\n", iterations);

    // allouer de la mémoire pour les tableaux
    double *k_means_x = (double *)malloc(sizeof(double) * numOfClusters);
    double *k_means_y = (double *)malloc(sizeof(double) * numOfClusters);

    if (k_means_x == NULL || k_means_y == NULL)
    {
        perror("malloc");
        exit(-1);
    }

    printf("Lecture des données d'entrée à partir du fichier...\n\n");

    FILE *fp = fopen("input.txt", "r");

    if (!fp)
    {
        perror("fopen");
        exit(-1);
    }

    // compter le nombre de lignes pour déterminer combien d'éléments
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

    printf("Il y a un nombre total de %d éléments dans le fichier.\n", numOfElements);

    // allouer de la mémoire pour un tableau de points de données
    double *data_x_points = (double *)malloc(sizeof(double) * numOfElements);
    double *data_y_points = (double *)malloc(sizeof(double) * numOfElements);
    int *k_assignment = (int *)malloc(sizeof(int) * numOfElements);

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

    printf("Exécution de l'algorithme k-means pour %d itérations...\n\n", iterations);
    for (int i = 0; i < numOfClusters; i++)
    {
        printf("K-means initiaux : (%f, %f)\n", k_means_x[i], k_means_y[i]);
    }

    clock_t start_time = clock();
    int count = 0;
    while (count < iterations) {
        assignerAuxClusters(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment, numOfElements, numOfClusters);
        calculerKmeans(k_means_x, k_means_y, data_x_points, data_y_points, k_assignment, numOfElements, numOfClusters);
        count++;
    }
    clock_t end_time = clock();

    printf("Execution time: %f seconds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);
    printf("--------------------------------------------------\n");
    printf("FINAL RESULTS:\n");
    for (int i = 0; i < numOfClusters; i++)
    {
        printf("Cluster #%d : (%f, %f)\n", i, k_means_x[i], k_means_y[i]);
    }
    printf("--------------------------------------------------\n");

    // Écrire les résultats finaux dans un fichier de sortie
    FILE *output_fp = fopen("results/output_seq.txt", "w");
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

    // désallouer la mémoire et nettoyer
    free(k_means_x);
    free(k_means_y);
    free(data_x_points);
    free(data_y_points);
    free(k_assignment);

    return 0;
}
