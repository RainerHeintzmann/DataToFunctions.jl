# DataToFunctions.jl
Represents (measured) data as a function, which supports affine and generally, matrix transformations for arrays. It is intended to be used as a tool for fitting data, where the fitting function is given by measured data.

It assumes the data as an Interpolation object (function), then it can transform the data to any arbitrary affine or matrix transformations.

What is important about this package is that it does not assume any pre-defined function as the data for the inverse modeling procedures.