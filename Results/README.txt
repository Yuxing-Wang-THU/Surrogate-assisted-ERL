Environment       Method             
Ant                     SPDERL-I           

HalfCheetah       SPDERL-I          

Hopper              SPDERL-I           

Walker               SPDERL-G           

Reacher               SERL-I            

Swimmer            SERL-G  
         
-----------------------------------------------------------------------------------------------------
Requirements:
torch                         1.7.0+cu101
numpy                         1.19.2
mujoco-py                     2.0.2.8
gym                           0.17.3
-----------------------------------------------------------------------------------------------------
Visualization & Evaluation:
python play.py -env=Ant-v2 -seed=3 -render -model_path=./Ant/ant_evo_net.pkl 
python play.py -env=HalfCheetah-v2 -seed=3 -render -model_path=./HalfCheetah/halfcheetah_evo_net.pkl
python play.py -env=Hopper-v2 -seed=3 -render -model_path=./Hopper/hopper_evo_net.pkl 
python play.py -env=Swimmer-v2 -seed=3 -render -model_path=./Swimmer/Swimmer_evo_net.pkl
python play.py -env=Walker2d-v2 -seed=3 -render -model_path=./Walker/walker_evo_net.pkl 
python play.py -env=Reacher-v2 -seed=3 -render -model_path=./Reacher/reacher_evo_net.pkl 
-----------------------------------------------------------------------------------------------------
Random tests:
python random_play.py -env=Ant-v2 -model_path=./Ant/ant_evo_net.pkl 
python random_play.py -env=HalfCheetah-v2 -model_path=./HalfCheetah/halfcheetah_evo_net.pkl
python random_play.py -env=Hopper-v2 -model_path=./Hopper/hopper_evo_net.pkl 
python random_play.py -env=Swimmer-v2 -model_path=./Swimmer/Swimmer_evo_net.pkl
python random_play.py -env=Walker2d-v2 -model_path=./Walker/walker_evo_net.pkl 
python random_play.py -env=Reacher-v2 -model_path=./Reacher/reacher_evo_net.pkl 
-----------------------------------------------------------------------------------------------------
