"""Conservative swept-volume boundary for mouth-held rope entry/exit."""
import numpy as np


def outside_rope_region(m,d,audit,rope_length,direction):
    if direction not in (-1,1):raise ValueError('direction must be -1 or +1')
    axis=float(np.mean([d.site('lavender/handle_tip').xpos[1],d.site('cream/handle_tip').xpos[1]]))
    # Any point on an inextensible length-L rope is within L/2 of the
    # midpoint of its endpoints. Include rope radius and attachment tolerance.
    radius=rope_length/2+float(np.max(m.geom_size[audit.rope_ids,0]))+.01
    limits=[]
    for geom in np.flatnonzero(audit.sky & ((m.geom_contype!=0)|(m.geom_conaffinity!=0))):
        if geom in audit.vertices:
            y=audit.vertices[geom]@d.geom_xmat[geom].reshape(3,3)[1]+d.geom_xpos[geom,1]
            limits.append(float(y.min() if direction==1 else y.max()))
        else:
            limits.append(float(d.geom_xpos[geom,1]-direction*m.geom_rbound[geom]))
    if not limits:raise ValueError('Robot has no collision geometry')
    return min(limits)>axis+radius if direction==1 else max(limits)<axis-radius


def local_rope_phase(m,d,audit):
    """Measured phase of the rope segment crossing this jumper's foot x-plane."""
    ids=audit.rope_ids
    half=d.geom_xmat[ids].reshape(-1,3,3)[:,:,2]*m.geom_size[ids,1,None]
    a,b=d.geom_xpos[ids]-half,d.geom_xpos[ids]+half
    feet=d.geom_xpos[audit.feet].mean(axis=0)
    delta=b[:,0]-a[:,0]
    f=np.divide(feet[0]-a[:,0],delta,out=np.full(len(ids),np.inf),where=np.abs(delta)>1e-8)
    valid=np.flatnonzero((f>=0)&(f<=1))
    if len(valid):
        points=a[valid]+f[valid,None]*(b[valid]-a[valid])
        point=points[np.argmin(np.abs(points[:,1]-feet[1]))]
    else:
        point=d.geom_xpos[ids[len(ids)//2]]
    axis_z=float(d.site_xpos[audit.handles,2].mean())
    return float(np.arctan2(point[1]-feet[1],axis_z-point[2]))
