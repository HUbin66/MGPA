from motif import frequentMotif


def get_motif(filename):
    #graph_path,out_graph_path,out_motif_path
    frequentMotif.get_motif_info("Data/"+filename+".txt", "Data/out_weight/"+filename+"2weight.txt",
                                 "Data/Motif/100"+'/'+filename+".txt")
