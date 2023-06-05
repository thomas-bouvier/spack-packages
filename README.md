# spack-packages

## `py-nvidia-dali`

The official DALI repository has the following dependency on `gast`:

```
'gast >= 0.2.1, <= 0.4.0'
```

This creates an issue with my Spack installation, as my environment can't be
concretized with `unify: true`. This custom version of DALI applies a path to
the DALI codebase to upgrade `gast >= 0.5`.