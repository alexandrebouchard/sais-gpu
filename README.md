## Interactive GPU prototyping on Sockeye


### Setup SSH

1. Setup a password manager to avoid having to type your CWL password every time
2. Make sure Sockeye connection sharing is setup. See [relevant section of Sockeye doc](https://confluence.it.ubc.ca/display/UARC/SSH+Connection+Sharing). The key info is that you should have the following in your laptop's `~/.ssh/config` file (`~` mean home directory):
```
Host s
    HostName sockeye.arc.ubc.ca
    User mycwl
    ControlMaster auto
    ControlPath ~/.ssh/ssh_mux_%h_%p_%r
```
3. Open the UBC VPN and a terminal window and login with `ssh s`, do the 2-factor auth. 
4. Put that terminal aside and don't touch it. As long as it is open, other terminals can be opened without the 2 step auth (required for VS Code remote to work).


### Setup remote VS Code

1. Open VS Code and click on the lower right icon `><` to start a remote VS Code running on Sockeye. 
2. For the hostname, just type `s`
3. Once you are in, use the terminal in VS Code, open it with apple/control with "`"


### Basic Sockeye setup

1. Create the directory `/scratch/st-alexbou-1/[your username]`
2. Download julia and install in your home dir, e.g. in `/home/[user]/bin/julia-1.xx.x`
3. Edit `~/.bash_profile` to add:
```
export PATH=$PATH:/home/alexbou/bin/julia-1.10.2/bin

export JULIA_PKG_DEVDIR=/scratch/st-alexbou-1/[your username]/
export JULIA_DEPOT_PATH=/scratch/st-alexbou-1/[your username]/depot
```
4. Test julia with `julia` then you will need this single package globally `] add Revise` (others are better to install in a project specific manner) then `exit()`
5. Edit `~/.bashrc` to add
```
alias j='julia --banner=no --load /home/[your username]/julia-start.jl --project=@. ' 
alias interact='salloc --time=3:00:00 --mem=16G --nodes=1 --ntasks=2 --account=st-alexbou-1'
alias ginteract='salloc --account=st-alexbou-1-gpu --partition=interactive_gpu --time=1:0:0 -N 1 -n 2 --mem=8G --gpus=1'
```
6. Create `~/julia-start.jl`
```
println("Active project: $(Base.active_project())")
println("Loading Revise...")
using Revise
```


### Setup this repo

1. Clone this repo, i.e. `cd /scratch/st-alexbou-1/[your username]`, `git clone https://github.com/alexandrebouchard/gpufun.git`, cd in it. 
2. Open two terminal in VS code:
    - One will be in the head node. Use for anything that needs network access (package install in particular). Open julia with `j`
    - One will be a GPU node that does not have internet access. In that terminal, type `ginteract` to start the GPU node, then `j`.
3. In the head node terminal, setup the repo with `] instantiate`


### Test your setup

In the GPU node's julia: `include("test.jl")`
