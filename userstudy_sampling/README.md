## Adaptively Learning the Crowd Kernel
Note that this code can not be just _plug and play_ and that it should be adapted to your user studies, or the data structures that you use.

There are two scripts in the folder:
- `ckl.py` is responsible to fit the answers you gather from the user studies to a new n-dimensional space that.
- `triplet_sampling.py` uses `ckl.py` to find the set of triplets that will
maximize the information gain in your two-alternative forced-choice (2AFC) user study.

This is an iterative algorithm where we sample the triplets that maximize the information gain (this also could be thought as sampling
the triplets that we have less information about), those triplets are answered by the
participants of the user study, with the new answers, we find again the triplets that maximize the information gain, and so on. 
This iterative process works as follows:
 - The first iteration the stimuli in the user study should be randomly sampled (we have no information yet in order to sample the pairs that
 give us the highest information gain.
 - After the first iteration.
   - Until we gain no information or for a fixed number of iterations:
     1. Gather all the previous answers in the user study
     2. Fit the CKL algorithm
     3. Sample the pairs that give you the most information gain

### How to adapt this code
To sample triplets for your user study, the script `triplet_sampling.py` should be modified. 
Most probably no modifications to `ckl.py` should be done.

In `triplet_sampling.py`:
- First set the paths to read the data of the user studies and output the sampled triplets
```python
answers_path = "Fill this"  # path where answers are stored
out_path = "Fill this"  # path to store the sampling ('data/sampling_iter_10.json')
nworkers = 10  # number of people doing the user study in each iteration
nquestions = 100  # number of triplets to be answer for each participant
```

- read all the answers from the user study until now (read the comments in the code to know how your data should look like)
```python
# read the input data of the test

# python list with each possible stimuli to sample in the user studies. 
# This could be just a list of paths where each path is a url to its corresponding stimulus.
input_data = # here goes the code to read the paths (or urls) to all the possible stimulus to sample  
nimages = len(input_data)

# get all the answered triplets in the user study until now

# answers is a np.array of integers with size (N, 3) containing all the
# triplets answered until now in the user studies (N). The triplets are
# stored as first the reference, and then the pair that the participants
# choose from. The array stores,  the class (as an integer) of each stimuli for each triplet.
# Note that the class is an integer with maximum value len(input_data). Therefore, answers[0, 0] 
# gives me the class of the reference stimulus of the triplet with index 0. if I access 
# input_data[answers[0, 0]], I am getting the url of that element.
# agreement is a np.array of size (N, 2) that stores the number of users
# that answered each pair. agreement[x, 0] corresponds to the number of
# users that have answered the stimuli answers[x, 1].
answers, agreement = # here goes the code to read the answers from the user study

# here we split the answers that have equal and different agreement.
# Equal agreement means that the same number of participants have choosen
# both pairs (agreement[x,0] == agreement[x,1]). Both keep the same format
# as the answers array. 
answers_equal, answers_diff = # here goes the code to split the answers
```

#### How to interpret the output
Then, after you run the script, a `json` file will be created on your specified `outpath`.
The file will have the following structure:
```json
{
  "hits_input_data":[...]
  "input_urls":[...]
}
```
where `hits_input_data` contains the triplets that maximize the information gain and that need to be sampled
in the next iteration of the user study. `input_urls` contains a list with each stimuli to be sampled.

`hits_input_data` contains a list with as many elements as `nworkers`, and each item in the list contains as many
triplets as `nquestions`. In particular, each item in the list is a dict with three keys. `R_input_urls_idx` corresponds
to the _reference_ stimulus of the triplet, while `A_input_urls_idx` and `B_input_urls_idx` is the stimuli that the participant
has to choose. To sample a triplet from this data structure we have to sample the same index for each key 
`(triplet = {R_input_urls_idx[0], A_input_urls_idx[0], B_input_urls_idx[0]})`
```json
"hits_input_data": [
    {
      "A_input_urls_idx": [67, ..],         
      "B_input_urls_idx": [29, ..],
      "R_input_urls_idx": [84, ..]
    },
    {..},
    {..},
    ..
 ]
 ```
 
`input_urls` is a list containing the information of each stimuli. In my case the _url_ of each image. This way, it is easy to access
the stimuli just by the integer of the class in `hits_input_data`.
  ```json
  "input_urls": [
    "http://giga.cps.unizar.es/~mlagunas/projects_data/merl_2AFC/havran-ennis-fullgeom/havran1_alum-bronze_ennis.jpg",
    "http://giga.cps.unizar.es/~mlagunas/projects_data/merl_2AFC/havran-ennis-fullgeom/havran1_alumina-oxide_ennis.jpg",
    "http://giga.cps.unizar.es/~mlagunas/projects_data/merl_2AFC/havran-ennis-fullgeom/havran1_aluminium_ennis.jpg",
    ..
    ]
  ```
  
  The idea would be to use this `.json` file to sample the triplets in your 2afc user study, and then repeat the iterative process explained
  above until your information gain in each iteration is almost zero.
  
  
  
 

