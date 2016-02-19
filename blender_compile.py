# 1455664595
import sys
import bpy

for i,v in enumerate(sys.argv):
    if v == '--':
        if len(sys.argv) <= i+1:
            exit(1)
        sys.path.insert(0, sys.argv[i+1])
        break
import export_spark_model
export_spark_model.save()

bpy.ops.wm.quit_blender()

