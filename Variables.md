# Environment Parameters #

Some GMAC policies can be chosen at run-time by setting environment variables. For instance, to override the default GMAC memory manager (Rolling), the user can set the `GMAC_MANAGER` variable in the following way:
```
GMAC_MANAGER=Lazy ./tests/gmacVecAdd
```
The GMAC code-base includes the necessary infrastructure to declare variables that take their value from environment variables. This code infrastructure is located in the `src/util` directory, and the GMAC constructor ensures that all programmer-defined environment-dependent variables get their correct value before GMAC starts executing.

## How to define a new Environment-Dependent Variable ##

Adding a new variable that takes its value from an environment value is straight-forward. You only need to add a `PARAM` declaration in the `src/util/Parameter.def` file. A `PARAM` definition looks like (note that `<options>` is optional):
```
PARAM(variableName, variableType, defaultValue, environmentName, <options>)
```
Note that **there is no `';'` mark at the end of each `PARAM` definition**. For instance, the variable that selects the memory manager is defined as follows:
```
PARAM(paramMemManager, char *, "Rolling", "GMAC_MANAGER")
```