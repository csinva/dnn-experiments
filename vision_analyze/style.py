def set_style():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.pylab as pylab
    from cycler import cycler
    plt.style.use('fivethirtyeight')
    label_size = 12
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size
    mpl.rcParams['lines.markersize'] = 10
    mpl.rcParams['lines.linewidth'] = 3.
    mpl.rcParams['grid.linewidth'] = 1.
    mpl.rcParams['legend.fontsize'] = label_size
    pylab.rcParams['xtick.major.pad']=3
    pylab.rcParams['ytick.major.pad']=3

    pylab.rcParams['figure.facecolor']='white'
    pylab.rcParams['axes.facecolor']='white'
    # mpl.rcParams['figure.figsize'] = [12, 10]
    # mpl.rcParams.keys()
    # Say, "the default sans-serif font is COMIC SANS"
    # mpl.rcParams['font.serif'] = 'Times New Roman'
    # # Then, "ALWAYS use sans-serif fonts"
    # mpl.rcParams['font.family'] = "Serif"
    
    from cycler import cycler
    alpha = 0.5
    to_rgba = mpl.colors.ColorConverter().to_rgba#
    color_list=[]
    for i, col in enumerate(plt.rcParams['axes.prop_cycle']):
        color_list.append(to_rgba(col['color'], alpha))
    plt.rcParams['axes.prop_cycle'] = cycler(color=color_list)
