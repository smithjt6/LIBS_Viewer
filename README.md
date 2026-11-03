The goals/Roadmap for this repo are: 

1. Implement a file loadng system, that is simply to use and can point at a directory and dynamically load files
    a. Can read and dyanmically/simply perform operations on TDMS files for LIBS spectroscopy
    b. Can read either paths or folders and dynamicall assign them to a file structure (Potenitaially dict --> K: filename, v: is data, or lists pointing to file data)
    c. can normalize data/accumulate if separate shots are provided, or accumulate over several 
2. Implement a plotting mechanism
    a. Can plot the data rapidly and plot either full spectrum or specific spectra wavelength windows
    b. Include LIBS Database + Custom LIBS wavelengths to allow for the 
3. Add in future ICP-MS integration (Agilent 7000)
    a. Similar to the 
3. Provide a nice UI with pages to visulaize data and simplify the data analytics process
    a. Provide an MCP/Tool for future LLM integration and reasoning
    b. Provide detailed docstring for future integrations (other sensors, techniques, etc.)
    c. Eventualy add in electrochemistry and corrosion solver + tools
