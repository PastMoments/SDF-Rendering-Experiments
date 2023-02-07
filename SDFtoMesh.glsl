#[compute]
#version 450

layout(local_size_x = 32) in;

/* ====================== OUPUT BUFFERS ====================== */

layout (std430, binding = 0, set = 0) restrict buffer outputInfo {
    uint vertex_count;
} output_info;

layout (std430, binding = 1, set = 0) restrict buffer vertexBuffer {
    float data[]; // godot can't do vec4
} vertex;

layout (std430, binding = 2, set = 0) restrict buffer normalBuffer {
    float data[]; // godot can't do vec4
} normal;



/* ====================== INPUT BUFFERS ====================== */

layout (std430, binding = 0, set = 1) restrict buffer inputInfo {
    float max_triangles;
	float base_radius;
	float smoosh;
	vec4 grid_size; 
} input_info;

layout (std430, binding = 1, set = 1) restrict buffer positionBuffer {
    float data[]; // godot can't do vec4
} position;





float get_sphere(vec3 p, vec3 center, float radius) {
	return length(p - center) - radius;
} 
float sminCubic( float a, float b, float k )
{
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*h*k*(1.0/6.0);
}

float map(vec3 p) {
	float s = 9999.0;
	for (int i = 0; i < position.data.length(); i+=3) {
		vec3 pos = vec3(position.data[i], position.data[i+1], position.data[i+2]);
		float a = get_sphere(p, pos, input_info.base_radius);
		if (a > 100.0) continue;
		s = sminCubic(a, s, input_info.smoosh);
	}
	return s;
}

vec3 get_normal(vec3 p) {
	const float NORMAL_PRECISION = 0.0005;
	float d = map(p);
	vec2 e = vec2(NORMAL_PRECISION, 0.0);
	vec3 n = vec3(d) - vec3(
		map(p - e.xyy),
		map(p - e.yxy),
		map(p - e.yyx));
	return normalize(n);
}

void addTriangle(vec3 a, vec3 b, vec3 c) {
	uint vertex_i = atomicAdd(output_info.vertex_count, 9);
	
	vec3 an = get_normal(a);
	vertex.data[vertex_i+0] = a.x;
	vertex.data[vertex_i+1] = a.y;
	vertex.data[vertex_i+2] = a.z;
	normal.data[vertex_i+0] = an.x;
	normal.data[vertex_i+1] = an.y;
	normal.data[vertex_i+2] = an.z;
	
	vec3 bn = get_normal(b);
	vertex.data[vertex_i+3] = b.x;
	vertex.data[vertex_i+4] = b.y;
	vertex.data[vertex_i+5] = b.z;
	normal.data[vertex_i+3] = bn.x;
	normal.data[vertex_i+4] = bn.y;
	normal.data[vertex_i+5] = bn.z;
	
	vec3 cn = get_normal(c);
	vertex.data[vertex_i+6] = c.x;
	vertex.data[vertex_i+7] = c.y;
	vertex.data[vertex_i+8] = c.z;
	normal.data[vertex_i+6] = cn.x;
	normal.data[vertex_i+7] = cn.y;
	normal.data[vertex_i+8] = cn.z;
}

vec3 placeVertex(vec3 p) {
	p += vec3(0.1);
	for (int i=0; i<4; ++i) {
		float d = map(p);
		vec3 norm = get_normal(p);
		float overshoot = 1.04 - step(0.0, d)*0.045;
		p -= (norm * d * overshoot);
		if (d < 0.0001 && d >= 0.0) {
			break;
		}
	}
	return p;
}

void addSurfaces(vec3 p) {
	float step = input_info.grid_size.w;
	vec2 e = vec2(step, 0.0);
	
	bool solid1 = map(p) > 0.0;
	bool solid2 = map(p + e.yyx) > 0.0;
	if (solid1 != solid2) {
		vec3 a = placeVertex(p - e.xxy);
		vec3 b = placeVertex(p - e.yxy);
		vec3 c = placeVertex(p - e.xyy);
		vec3 d = placeVertex(p - e.yyy);
		if (!solid2) {
			addTriangle(a,b,c);
			addTriangle(c,b,d);
		} else {
			addTriangle(c,b,a);
			addTriangle(d,b,c);
		}
	}

	solid2 = map(p + e.yxy) > 0.0;
	if (solid1 != solid2) {
		vec3 a = placeVertex(p - e.xyx);
		vec3 b = placeVertex(p - e.yyx);
		vec3 c = placeVertex(p - e.xyy);
		vec3 d = placeVertex(p - e.yyy);
		if (!solid1) {
			addTriangle(a,b,c);
			addTriangle(c,b,d);
		} else {
			addTriangle(c,b,a);
			addTriangle(d,b,c);
		}
	}
	
	
	solid2 = map(p + e.xyy) > 0.0;
	if (solid1 != solid2) {
		vec3 a = placeVertex(p - e.yxx);
		vec3 b = placeVertex(p - e.yyx);
		vec3 c = placeVertex(p - e.yxy);
		vec3 d = placeVertex(p - e.yyy);
		if (!solid2) {
			addTriangle(a,b,c);
			addTriangle(c,b,d);
		} else {
			addTriangle(c,b,a);
			addTriangle(d,b,c);
		}
	}
}


void main() {
	uvec3 worker_size = gl_NumWorkGroups * gl_WorkGroupSize;
	uint worker_i = gl_GlobalInvocationID.x;
	
	uint total_num_workers = worker_size.x * worker_size.y * worker_size.z;
	
	float step = input_info.grid_size.w;
	uvec3 grid_size = uvec3(input_info.grid_size.xyz);
	uvec3 total_compute_space = uvec3(grid_size/vec3(step));
	uint total_num_spaces = total_compute_space.x * total_compute_space.y * total_compute_space.z;
	
	uint spaces_per_worker = total_num_spaces/total_num_workers;
	uint starting_i = spaces_per_worker*worker_i;
	for (uint i=0; i<spaces_per_worker; ++i) {
		uint cur_i = (starting_i + i);
		if (cur_i > total_num_spaces) return;
		float grid_x = mod(cur_i, total_compute_space.x);
		float grid_y = mod(floor(cur_i/total_compute_space.x), total_compute_space.y);
		float grid_z = cur_i/(total_compute_space.y*total_compute_space.x);
		vec3 grid_cur = (vec3(grid_x, grid_y, grid_z) - total_compute_space/2.0) * step;
		addSurfaces(grid_cur);
	}
}


