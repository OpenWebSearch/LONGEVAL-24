# IR Lab in Padua 2024: Building Search Engines for LongEval'24 at CLEF 2024

This repository serves as starter for the IR lab in Padua 2024 for participation at [LongEval 2024](https://clef-longeval.github.io/) at [CLEF 2024](https://clef2024.imag.fr/) using [TIRA](https://www.tira.io/)/[TIREx](https://www.tira.io/tirex).

We recommend that you work using [dev containers with Docker](https://code.visualstudio.com/docs/devcontainers/containers) to simplify reproducible submissions to TIRA.


## Developing in Dev Containers

A dev container (please find a suitable installation instruction [here](https://code.visualstudio.com/docs/devcontainers/containers)) allows you to directly work in the prepared Docker container so that you do not have to install the dependencies (which can sometimes be a bit tricky).

To develop with dev containers, please:

- Install [VS Code](https://code.visualstudio.com/download) and [Docker](https://docs.docker.com/engine/install/) on your machine
- Clone this repository: `git clone ...`
- Open this repository in VS Code, click on "Open in Container", shortly verify that you are in the container.


## Repository Structure

This repository contains a set of directories as inspiration on how to get started.
As one of your first steps, you can look into each directory and delete the directories that you do not want to use (if you later change your mind, you can still get everything from the git history).

The directories are:

- [ir-datasets](ir-datasets): Contains the code for parsing queries and documents. If you want to focus your work on new/different ways to parse the queries/documents (e.g., using the french text or a combination of english/french), this would be your starting point.
- [java-retrieval](java-retrieval): Contains code for retrieval pipelines that use java with Maven.
- [pyterrier-retrieval](pyterrier-retrieval): Contains code for retrieval pipelines that use python with [PyTerrier](https://github.com/terrier-org/pyterrier).


## Resources and Links

- Please register to TIRA at: [https://www.tira.io/task-overview/ir-lab-padua-2024](https://www.tira.io/task-overview/ir-lab-padua-2024)
- A dashboard overviewing retrieval components available in [TIRA](https://www.tira.io/)/[TIREx](https://www.tira.io/tirex) with tutorials on how you could re-use them: [https://tira-io.github.io/teaching-ir-with-shared-tasks/](https://tira-io.github.io/teaching-ir-with-shared-tasks/)
- An [ir_datasets Browser](https://tira-io.github.io/ir-dataset-browser) to inspect public datasets from [https://ir-datasets.com/](https://ir-datasets.com/) with systems submitted to [TIRA](https://www.tira.io/)/[TIREx](https://www.tira.io/tirex)
- The template repository for this repo (in case if you have improvements for the next generation :)): [https://github.com/tira-io/ir-lab-padua-2024](https://github.com/tira-io/ir-lab-padua-2024)

Everything is open for modification and we would be happy about ideas and pull requests!


## Assistance and Help

**Idea:** [Maik](https://www.tira.io/u/maik_froebe) could do an early review of your repository, e.g., after the first two weeks, to give feedback on the dockerization to prepare an early submission of your baseline upon which you subsequently can improve upon.

If you have any questions or problems, please do not hesitate to contact [Maik](https://www.tira.io/u/maik_froebe) via mail or a [direct message in TIRA](https://www.tira.io/u/maik_froebe), or post your question on [the forum](https://www.tira.io/c/ir-lab/).
