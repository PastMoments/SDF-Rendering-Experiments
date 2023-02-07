@tool
extends "./sdf_item.gd"


@export var radius: float:
	get:
		return _data.params[SDF.PARAM_RADIUS].value
	set(r):
		radius = r # Useless but doing it anyways
		_set_param(SDF.PARAM_RADIUS, r)
@export var thickness: float:
	get:
		return _get_param(SDF.PARAM_THICKNESS)
	set(r):
		thickness = r # Useless but doing it anyways
		_set_param(SDF.PARAM_THICKNESS, r)


func _init():
	_data = SDF.SceneObject.new(SDF.SHAPE_TORUS)
