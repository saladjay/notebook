import os
import generate_example_dataset.create_datasets as create_datasets
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_classes", type=int, default=15)
    parser.add_argument("--nb_tags", type=int, default=10)
    parser.add_argument("--nb_datasets", type=int, default=5)
    parser.add_argument("--nb_images", type=int, default=1000)
    parser.add_argument("--nb_labels", type=int, default=10000)
    parser.add_argument("--db_file", type=str, default="example_dataset.db")
    args = parser.parse_args()
    create_datasets.create_example_dataset(args.db_file, args.nb_classes, args.nb_tags, args.nb_datasets, args.nb_images, args.nb_labels)