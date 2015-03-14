Installing GMAC


# Install from Mercurial #

These instructions will download, compile and install GMAC into your system. Please, consider using pre-compiled packages if you just want to use GMAC.

> ## Getting the code ##
> We are now delivering source code packages, check the [downloads page](http://code.google.com/p/adsm/downloads/list), and go directly to the [Compiling](Install#Compiling_the_code.md) section.

> GMAC uses mercurial as control version system to manage the source code and documentation. You need mercurial version 1.3 installed in your system to download the GMAC source code. You can check if mercurial is installed typing:
```
 hg --version 
```
> > ### The Easy Way ###
> > The ADSM project has a default commodity repository that includes the main packages that form GMAC. This default repository includes the latest /stable/  version of GMAC. If you need the latest (/unstable/) features of GMAC, use the [the hard way](Install#The_Hard_Way.md) of getting the source code.
> > You need to clone the GMAC repository in your system. This operation might take some time depending on your system and network connection.
```
  hg clone https://adsm.googlecode.com/hg/ adsm
  (cd adsm; ./setup.sh)
```
> > ### The Hard Way ###
> > GMAC modules are stored using separate mercurial repositories. This source code management policy allows us to keep a clean packaging system, where updates for a single module do not require updating the other GMAC modules. To be able to compile GMAC you need to download the source code for all repositories using a fixed directory hierarchy.
> > To download the GMAC repository you have to execute:
```
  hg clone https://libgmac.adsm.googlecode.com/hg/ adsm/libgmac
  (cd adsm/libgmac; ./setup.sh)
```


> ## Compiling the code ##
> The `setup.sh` script executed when downloading the code generates the configurations scripts required to compile the code. There are many possible ways to compile GMAC, but we recommend using a separate `build` directory instead of compiling the code on the same directory than the source code. Compiling the code involves two steps, configuring the source code with the desired compilation flags, and compiling the code. The following code shows how to perform these two steps.
```
  mkdir -p adsm/build
  cd adsm/build
  ../libgmac/configure --enable-tests
  make
```

> ## Installing GMAC ##
The configuration scripts sets `/usr/` as the default installation path. If you want to install GMAC on a different path, please use the `--prefix` flag in the configuration script. Before installing GMAC we recommend running some test programs to ensure that GMAC was compiled correctly
```
  ./gmacVecadd
  ./gmacThreadVecAdd
  ./gmacStencil
```
All these tests will show the execution time and the error. An error value different from zero (or the test refusing to execute) would mean that something went wrong during the compilation. Once you have tested your compilation, you can install the GMAC library and header files by executing:
` make install `

> Congratulations, GMAC is ready to be used in your system