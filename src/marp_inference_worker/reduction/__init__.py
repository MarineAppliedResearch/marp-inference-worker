# __init__.py
# Created: 2026-09-09
# Author: Isaac Travers
#
# Package marker for the reduction package of the MARP Inference Worker.
# Import from the modules inside it rather than re-exporting here, so a module
# that pulls in torch is only loaded by something that needs torch.
