@tool
extends Path3D


# Called when the node enters the scene tree for the first time.
func _ready():
	print(curve.get_baked_tilts())


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	pass
