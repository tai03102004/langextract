| Parameter | Description |
| --- | --- |
| data | Input SAS data set (required); its name must not start with two underscores, and it must not contain variables whose names start with two underscores (in particular, “__g”, “__t”, and “__status”) |
| time | Survival time variable (required) |
| timeunit | Time units of the survival time variable; default is “years” |
| status | Survival status variable (required) |
| censval | Censoring status value(s) to be used in PROC LIFETEST; default is “0”; use the %str() function to specify more than 1 value |
| group | Covariate which distinguishes the two groups of interest (required) |
| gvalue1 | Numerical value of first group of interest (required) |
| gvalue2 | Numerical value of second group of interest (required) |
| grouplbl | Label of group variable (optional) |
| gvallbl1 | Label of first group value (optional) |
| gvallbl2 | Label of second group value (optional) |
| alpha | 100‐alpha % is the pointwise two‐sided confidence level; default for alpha is “5” |
| boot | Number of bootstrap replications for computing confidence bands; default is “2000”, minimum is “100” |
| bundle | Number of bootstrap replications shown in figures; default is “200” |
| seedval | Bootstrap random number seed; default is “0” |
