include("./config.jl")

targets = keys(targetsDict)


function make_slurm_jobs(;script_to_run="5a__hpo_DecisionTreeRegressor.jl", basename="5a_hpo_", n_tasks=8, datapath="/media/john/HSDATA/datasets/Full", outpath="/media/john/HSDATA/analysis_full")
    for target ∈ targets
        job_name = basename*"$(target)"

        file_text = """
        #!/bin/bash

        #SBATCH     --job-name=$(job_name)
        #SBATCH     --output=$(job_name).out
        #SBATCH     --error=$(job_name).err
        #SBATCH     --nodes=1
        #SBATCH     --ntasks=1
        #SBATCH     --cpus-per-task=$(n_tasks)   # number of threads for multi-threading
        #SBATCH     --time=2-00:00:00
        #SBATCH     --mem=30G
        #SBATCH     --mail-type=ALL
        #SBATCH     --mail-user=jxw190004@utdallas.edu
        #SBATCH     --partition=normal

        julia --threads \$SLURM_CPUS_PER_TASK --project=../../ $(script_to_run) -T $(target) -d $(datapath) -o $(outpath)
        """



        open(basename*String(target)*".slurm", "w") do f
            println(f, file_text)
        end


    end
end


# make_slurm_jobs(;
#                 script_to_run="5a__hpo_DecisionTreeRegressor.jl",
#                 basename="5a__hpo_",
#                 n_tasks=4,
# 		datapath="/scratch/jwaczak/data/datasets/Full",
# 		outpath="/scratch/jwaczak/data/analysis_full",
#                 )


# make_slurm_jobs(;
#                 script_to_run="5b__hpo_RandomForestRegressor.jl",
#                 basename="5b_hpo_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/Full",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )


# make_slurm_jobs(;
#                 script_to_run="5d__hpo_KNNRegressor.jl",
#                 basename="5d__hpo_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/Full",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )


# make_slurm_jobs(;
#                 script_to_run="5c__hpo_XGBoostRegressor.jl",
#                 basename="5c_hpo_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/Full",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )


# make_slurm_jobs(;
#                 script_to_run="5e__hpo_EvoTreeRegressor.jl",
#                 basename="5e_hpo_",
#                 n_tasks=2,
# 		            datapath="/scratch/jwaczak/data/datasets/Full",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )


# make_slurm_jobs(;
#                 script_to_run="5f__hpo_LGBMRegressor.jl",
#                 basename="5f_hpo_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/Full",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )


# make_slurm_jobs(;
#                 script_to_run="7__superlearner.jl",
#                 basename="7_sl_",
#                 n_tasks=4,
# 		            datapath="/scratch/jwaczak/data/datasets/Full",
# 		            outpath="/scratch/jwaczak/data/analysis_full",
#                 )

make_slurm_jobs(;
                script_to_run="8__evaluate_superlearners.jl",
                basename="8_eval_11-23_",
                n_tasks=4,
		            datapath="/scratch/jwaczak/data/datasets/11-23",
		            outpath="/scratch/jwaczak/data/analysis_full",
                )

make_slurm_jobs(;
                script_to_run="8__evaluate_superlearners.jl",
                basename="8_eval_12-09_",
                n_tasks=4,
		            datapath="/scratch/jwaczak/data/datasets/12-09",
		            outpath="/scratch/jwaczak/data/analysis_full",
                )

make_slurm_jobs(;
                script_to_run="8__evaluate_superlearners.jl",
                basename="8_eval_12-10_",
                n_tasks=4,
		            datapath="/scratch/jwaczak/data/datasets/12-10",
		            outpath="/scratch/jwaczak/data/analysis_full",
                )


