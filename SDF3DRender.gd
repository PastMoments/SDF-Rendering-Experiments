@tool
extends MeshInstance3D

class_name SDF3DRender

@onready var _is_ready := true # a little hacky

enum renderMode {Raymarcher, Mesher}

var arrayMesh: ArrayMesh = ArrayMesh.new()
var raymarchBox: BoxMesh = BoxMesh.new()

@export var render_mode: renderMode = renderMode.Raymarcher:
	set(mode):
		render_mode = mode
		print("setting render mode")
		if mode == renderMode.Raymarcher:
			await updateRaymarchBox(raymarchBox)
			self.mesh = raymarchBox			
		elif mode == renderMode.Mesher:
			await updateMesh(arrayMesh)
			self.mesh = arrayMesh

	
@export var bounding_box_size = Vector3i(20,20,20):
	set(a): 
		bounding_box_size = a
		await updateRaymarchBox(raymarchBox)
		await updateMesh(arrayMesh)
@export var base_radius = 0.25:
	set(a):
		base_radius = a
		await updateRaymarchBox(raymarchBox)
		await updateMesh(arrayMesh)
@export var smoosh = 2.0:
	set(a):
		smoosh = a
		await updateRaymarchBox(raymarchBox)
		await updateMesh(arrayMesh)
@export_range(0.1, 0.6) var step = 0.5: # TODO: make work with larger steps
	set(a):
		step = a
		await updateMesh(arrayMesh)
@export var max_triangles = 262144:
	set(a):
		max_triangles = a
		await updateMesh(arrayMesh)

func _on_path_3d_curve_changed():
	await updateRaymarchBox(raymarchBox)
	await updateMesh(arrayMesh)

func updateRaymarchBox(box):
	if not self._is_ready:
		await self.ready
	if render_mode != renderMode.Raymarcher:
		return
	print("updating raymarch box")
	box.size = bounding_box_size
	box.material = preload("res://raymarcher.material")
	box.material.set_shader_param("bounding_box_size", Vector3(bounding_box_size.x, bounding_box_size.y, bounding_box_size.z))
	box.material.set_shader_param("base_radius", base_radius)
	box.material.set_shader_param("smoosh", smoosh)
	var path = $Path3D
	var points = path.curve.get_baked_points() 
	box.material.set_shader_param("points", points)
	box.material.set_shader_param("numpoints", points.size())
	box.material.set_shader_param("local_trans", global_transform)
	return box

var meshUpdateMutex: Mutex = Mutex.new()
func updateMesh(old_mesh):
	if not self._is_ready:
		await self.ready
	if render_mode != renderMode.Mesher:
		return
	if meshUpdateMutex.try_lock() != OK:
		print("updateMesh is busy")
		return
		
	print("updating mesh")

	# Create a local rendering device. 
	var rd := RenderingServer.create_local_rendering_device()
	# Load GLSL shader
	var shader_file := preload("res://SDFtoMesh.glsl")
	var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()
	var shader := rd.shader_create_from_spirv(shader_spirv)
		
	var output_info := PackedFloat32Array([0.0])
	# 0 - number of vertices
	var vertices := PackedVector3Array()
	vertices.resize(max_triangles*3)
	var normals := PackedVector3Array()
	normals.resize(max_triangles*3)
	
	var input_info := PackedFloat32Array([max_triangles, base_radius, smoosh, 0, bounding_box_size.x, bounding_box_size.y, bounding_box_size.z, step])
	
	var points: PackedVector3Array = get_node("Path3D").curve.get_baked_points()
	# TODO: a better way of sending input data
	
	# outputs
	var output_info_bytes := output_info.to_byte_array()
	var output_info_buffer := rd.storage_buffer_create(output_info_bytes.size(), output_info_bytes)
	var output_info_uniform := RDUniform.new()
	output_info_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	output_info_uniform.binding = 0
	output_info_uniform.add_id(output_info_buffer)
	
	var vertices_bytes := vertices.to_byte_array()
	var vertices_buffer := rd.storage_buffer_create(vertices_bytes.size(), vertices_bytes)
	var vertices_uniform := RDUniform.new()
	vertices_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	vertices_uniform.binding = 1
	vertices_uniform.add_id(vertices_buffer)
	
	var normals_bytes := normals.to_byte_array()
	var normals_buffer := rd.storage_buffer_create(normals_bytes.size(), normals_bytes)
	var normals_uniform := RDUniform.new()
	normals_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	normals_uniform.binding = 2
	normals_uniform.add_id(normals_buffer)
	
	var output_set := rd.uniform_set_create([output_info_uniform, vertices_uniform, normals_uniform], shader, 0)
	
	#  inputs
	var input_info_bytes := input_info.to_byte_array()
	var input_info_buffer := rd.storage_buffer_create(input_info_bytes.size(), input_info_bytes)
	var input_info_uniform := RDUniform.new()
	input_info_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	input_info_uniform.binding = 0
	input_info_uniform.add_id(input_info_buffer)
	
	var points_bytes := points.to_byte_array()
	var points_buffer := rd.storage_buffer_create(points_bytes.size(), points_bytes)
	var points_uniform := RDUniform.new()
	points_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	points_uniform.binding = 1
	points_uniform.add_id(points_buffer)
	
	var input_set := rd.uniform_set_create([input_info_uniform, points_uniform], shader, 1)
	
	# Create a compute pipeline
	var pipeline := rd.compute_pipeline_create(shader)
	var compute_list := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, output_set, 0)
	rd.compute_list_bind_uniform_set(compute_list, input_set, 1)
	rd.compute_list_dispatch(compute_list, 1024, 1, 1)
	rd.compute_list_end()
	
	# Submit to GPU and wait for sync
	rd.submit()
	rd.sync()
	
	# Read back the data from the buffers
	output_info = rd.buffer_get_data(output_info_buffer).to_float32_array()
	
	var output_vertices_bytes := rd.buffer_get_data(vertices_buffer)
	var output_normals_bytes := rd.buffer_get_data(normals_buffer)
	var output_vertices = byte_array_to_vector3_array(output_vertices_bytes).slice(output_info[0])
	var output_normals = byte_array_to_vector3_array(output_normals_bytes).slice(output_info[0])
	
	var arrays = []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = output_vertices
	arrays[Mesh.ARRAY_NORMAL] = output_normals

	# Create the Mesh.
	old_mesh.clear_surfaces()
	old_mesh.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	
	meshUpdateMutex.unlock()

# TODO: replace when bug fixed
func byte_array_to_vector3_array(byte_arr: PackedByteArray) -> PackedVector3Array:
	var vertices_floats = byte_arr.to_float32_array()
	var output := PackedVector3Array() # TODO: cast directly when bug fixed
	output.resize(vertices_floats.size()/3)
	for i in output.size():
		var vi = i*3
		output[i] = Vector3(vertices_floats[vi], vertices_floats[vi+1], vertices_floats[vi+2])
	return output
	
# Called when the node enters the scene tree for the first time.
func _ready():
	render_mode = render_mode # update current render mode stuff

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta):
	if not self._is_ready:
		await self.ready
	if render_mode == renderMode.Raymarcher:
		raymarchBox.material.set_shader_param("local_trans", global_transform)


