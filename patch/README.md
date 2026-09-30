# patch/

Files here are copied over the installed `tensorfold` package when the image is built with
`TF_LOCAL=1 scripts/build.sh` (see the Dockerfile). It is a **list of files**, never a tree: an overlay of a
whole working tree replaces every file that tree lacks at whatever revision it happens to carry, which is how
an older `vision/config.py` once shipped and refused the checkpoint at launch. Keep it to files that differ,
each derived from the revision `TF_REF` installs.
