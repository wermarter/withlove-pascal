uses process;
var proc:tprocess;
begin
proc:=tprocess.create(nil);
proc.options:=[ponoconsole];
proc.priority:=ppRealTime;
proc.executable:='C:\Program Files\CCleaner\ccleaner64.exe';
proc.parameters.add('/AUTO');
proc.execute;
proc.free
end.
