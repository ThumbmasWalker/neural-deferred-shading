import sys
import os

def main():
    sys.path.append("nds\\utils")
    from mesh import export_spheres
    export_spheres(range(8), "mesh_files")
if __name__ == '__main__':
    main()