
include("./config.jl")
targets = keys(targets_dict)


function make_slurm_jobs(;script_to_run="5a__hpo_DecisionTreeRegressor.jl", basename="5a-", n_tasks=8, datapath="/media/john/HSDATA/datasets/Full", outpath="/media/john/HSDATA/analysis_full")

    #job_name = basename*"$(target)"
    job_name = basename

    file_text = """
    #!/bin/bash

    #SBATCH     --job-name=$(job_name[1:end-1])
    #SBATCH     --array=1-29
    #SBATCH     --output=$(job_name)%A_%a.out
    #SBATCH     --error=$(job_name)%A_%a.err
    #SBATCH     --nodes=1
    #SBATCH     --ntasks=1
    #SBATCH     --cpus-per-task=$(n_tasks)   # number of threads for multi-threading
    #SBATCH     --time=2-00:00:00
    #SBATCH     --mem=30G
    #SBATCH     --mail-type=ALL
    #SBATCH     --mail-user=jxw190004@utdallas.edu
    #SBATCH     --partition=normal

    julia --threads \$SLURM_CPUS_PER_TASK --project=../../ $(script_to_run) -i \$SLURM_ARRAY_TASK_ID -d $(datapath)
    """

    open(basename[1:end-1]*".slurm", "w") do f
        println(f, file_text)
    end
end



make_slurm_jobs(;
                script_to_run="5a__hpo-DecisionTreeRegressor.jl",
                basename="5a-",
                n_tasks=4,
                datapath="/scratch/jwaczak/data/robot-team/supervised",
                )


make_slurm_jobs(;
                script_to_run="5b__hpo-RandomForestRegressor.jl",
                basename="5b-",
                n_tasks=4,
                datapath="/scratch/jwaczak/data/robot-team/supervised",
                )


make_slurm_jobs(;
                script_to_run="5c__hpo-XGBoostRegressor.jl",
                basename="5c-",
                n_tasks=4,
                datapath="/scratch/jwaczak/data/robot-team/supervised",
                )


make_slurm_jobs(;
                script_to_run="5d__hpo-KNNRegressor.jl",
                basename="5d-",
                n_tasks=4,
                datapath="/scratch/jwaczak/data/robot-team/supervised",
                )


make_slurm_jobs(;
                script_to_run="5e__hpo-EvoTreeRegressor.jl",
                basename="5e-",
                n_tasks=4,
                datapath="/scratch/jwaczak/data/robot-team/supervised",
                )


make_slurm_jobs(;
                script_to_run="5f__hpo-LGBMRegressor.jl",
                basename="5f-",
                n_tasks=4,
                datapath="/scratch/jwaczak/data/robot-team/supervised",
                )


make_slurm_jobs(;
                script_to_run="5g__hpo-CatBoostRegressor.jl",
                basename="5g-",
                n_tasks=4,
                datapath="/scratch/jwaczak/data/robot-team/supervised",
                )


# make_slurm_jobs(;
#                 script_to_run="6__train-superlearner.jl",
#                 basename="6-",
#                 n_tasks=4,
#                 datapath="/scratch/jwaczak/data/robot-team/supervised",
#                 )


# make_slurm_jobs(;
#                 script_to_run="8__evaluate_superlearners.jl",
#                 basename="8_eval_11-23_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/11-23",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )

# make_slurm_jobs(;
#                 script_to_run="8__evaluate_superlearners.jl",
#                 basename="8_eval_12-09_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/12-09",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )

# make_slurm_jobs(;
#                 script_to_run="8__evaluate_superlearners.jl",
#                 basename="8_eval_12-10_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/12-10",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )


