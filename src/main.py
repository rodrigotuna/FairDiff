from model.fairdiff import FairDiff
from datasets.dataset import SampledDataset
from datasets.fair_rw import FairRW


def main():
    #model = FairDiff()
    print("Done")
    data = SampledDataset("Cora", FairRW(), 100)
    print(data[1].edge_index)
    


if __name__ == "__main__":
    main()