class position:
	rx, ry, rz = 0, 0, 0

	def pos_init(self, rx, ry, rz):
		self.rx = rx
		self.ry = ry
		self.rz = rz

class velocity:
	vx, vy, vz = 0, 0, 0

	def vel_init(self, vx, vy, vz):
		self.vx = vx
		self.vy = vy
		self.vz = vz

class lj_force:
	fx, fy, fz = 0, 0, 0

	def ljf_init(self, fx, fy, fz):
		self.fx = fx
		self.fy = fy
		self.fz = fz

class common3D:
	x, y, z = 0, 0, 0