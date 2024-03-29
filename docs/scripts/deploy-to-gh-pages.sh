#!/bin/bash
# Based off instructions from:
# https://gist.github.com/domenic/ec8b0fc8ab45f39403dd

set -o errexit -o nounset

PACKAGE="pybreakpoints"

DOCS=$(dirname $(readlink -f $0))/../
APIDOC="${DOCS}/source/${PACKAGE}"

SOURCE=_build

REPO=$(git config remote.origin.url)
SSH_REPO=${REPO/https:\/\/github.com\//git@github.com:}
REV=$(git rev-parse --short HEAD)

DST_BRANCH=gh-pages

# Determine SRC_BRANCH
set +u
if [ "$TRAVIS" == "true" ]; then
    set -u

    echo "On Travis CI"
    if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then
        echo "Not building docs for PR"
        exit 0
    fi

    # Override what git says
    if [ "$TRAVIS_PULL_REQUEST" == "false" ]; then
        SRC_BRANCH=$TRAVIS_BRANCH
    else
        SRC_BRANCH=$TRAVIS_PULL_REQUEST_BRANCH
    fi

    git config --global user.email $COMMIT_AUTHOR_EMAIL
    git config --global user.name $COMMIT_AUTHOR_NAME
else
    SRC_BRANCH=$(git rev-parse --abbrev-ref HEAD)
fi
set -u

echo "Building docs for branch: $SRC_BRANCH"

DEST=$SOURCE/$SRC_BRANCH/

# START
cd $DOCS/

# Clean
rm -rf build
rm -rf $SOURCE

# Create branch directory and grab Git repo
echo "Cloning repo: $SSH_REPO"
echo "git clone $SSH_REPO $SOURCE/"
git clone $SSH_REPO $SOURCE/
cd $SOURCE/
git checkout $DST_BRANCH || git checkout --orphan $DST_BRANCH
cd $DOCS/
rm -rf $DEST || exit 0

# Generate API doc
sphinx-apidoc -f -e -o $APIDOC ../${PACKAGE}/

# Build docs
make html
rm -rf $DEST
mkdir -p $DEST
cp -R build/html/* $DEST

# If there's test coverage results, add it in!
HTMLCOV=../htmlcov
if [ -d $HTMLCOV ]; then
    # Copy to new directory name since "htmlcov" is in gitignore
    echo "Copying coverage HTML report to docs"
    cp -vR $HTMLCOV $DEST/coverage
    # Generate a badge
    badge=$DEST/coverage_badge.svg
    if [ -f ../badge.svg ]; then
        echo "Copying coverage badge to $badge"
        cp -v ../badge.svg $badge
    else
        echo "Cannot find coverage badge"
        wget https://img.shields.io/badge/docs-error-lightgrey.svg -O $badge
    fi
fi

# Commit and push to GH
cd $SOURCE/
echo "Adding files from $SRC_BRANCH"
git add -v $SRC_BRANCH
git commit -m "Rebuild $DST_BRANCH docs on $SRC_BRANCH: ${REV}"
git push origin HEAD:$DST_BRANCH
